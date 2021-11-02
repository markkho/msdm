import torch
import numpy as np
import cvxpy as cp

from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.pomdp import TabularPOMDP
from msdm.core.problemclasses.pomdp.finitestatecontroller import StochasticFiniteStateController
from msdm.core.algorithmclasses import Learns, Result

from msdm.algorithms.fscgradientascent import stochastic_fsc_policy_evaluation_exact


def with_new_node(fsc_action, fsc_state, new_action, new_state):
    '''
    Returns a copy of the supplied FSC with an additional node.
    '''
    ncontroller, nactions = fsc_action.shape
    nobs = fsc_state.shape[2]
    assert fsc_state.shape == (ncontroller, nactions, nobs, ncontroller)
    assert new_action.shape == (nactions,)
    assert new_state.shape == (nactions, nobs, ncontroller+1)

    nfsc_action = np.concatenate([fsc_action, new_action[None, :]])
    nfsc_state = np.concatenate([
        np.concatenate([
            fsc_state,
            # Add zero probability edges in towards new node
            np.zeros(fsc_state.shape[:-1]+(1,)),
        ], axis=-1),
        # Add new node's outward edges
        new_state[None],
    ], axis=0)

    assert nfsc_action.shape == (ncontroller+1, nactions)
    assert nfsc_state.shape == (ncontroller+1, nactions, nobs, ncontroller+1)

    return nfsc_action, nfsc_state


def propose_escape_node(pomdp, belief, state_controller_value):
    '''
    We return a proposal for a node to help escape a local minima. We pick
    the best node for this belief state.
    '''

    nactions, nstates, nobs = pomdp.observation_matrix.shape
    ncontroller = state_controller_value.shape[0]
    assert state_controller_value.shape == (ncontroller, nstates)

    # Computing expected immediate reward
    reward = belief @ pomdp.state_action_reward_matrix
    # Variables for best action thus far
    max_v = -float('inf')
    max_as = None
    # Tracking observation strategy. We hold onto this across actions
    # since unused actions are no-ops, and since this gives us something
    # that is a valid probability distribution for all inputs.
    observation_strategy = np.zeros((nactions, nobs, ncontroller+1))

    for ai, a in enumerate(pomdp.action_list):
        # initialize value to the immediate reward
        v = reward[ai]

        for oi, s_ns_o_prob in enumerate(pomdp.predictive_observation_vec(belief, ai)):
            next_belief = pomdp.state_estimator_vec(belief, ai, oi)

            # Given the belief that results from (a, o), we compute a value over our existing nodes.
            controller_value = state_controller_value @ next_belief
            # We pick the node that's best for this (a, o)
            nn = np.argmax(controller_value)
            observation_strategy[ai, oi, nn] = 1
            # and sum in the value to our running tally
            v += pomdp.discount_rate * s_ns_o_prob * controller_value[nn]

        # Checking to see if this action has maximum value.
        if v > max_v:
            max_v = v
            action_strategy = np.zeros(nactions)
            action_strategy[ai] = 1
            max_as = action_strategy

    current_v = np.max(state_controller_value @ belief)
    assert np.isclose(max_v, current_v) or max_v > current_v, 'is this true?????'

    # Assemble result
    r = Result(
        current_v=current_v,
        improved=max_v > current_v and not np.isclose(max_v, current_v),
        max_v=max_v,
        action_strategy=max_as,
        observation_strategy=observation_strategy,
    )
    r.add_to_fsc = lambda fsc_action, fsc_state: with_new_node(fsc_action, fsc_state, r.action_strategy, r.observation_strategy)
    return r

def improve_node(pomdp, V, node, *, solver=cp.ECOS, verbose=False):
    '''
    This implements a linear program to find a combination of backed-up FSC nodes that dominate
    an existing node. It implements the efficient version described in Table 4 of [1]. In particular,
    we follow the suggestion in a footnote to compute c_a based on the c_{a,n_z} values, leading to a
    total parameter size of |A||N||Z|+1.

    Since we compute c_a based on c_{a,n_z}, we can make a small simplification to the LP in Table 4:
    - We drop the constraint that c_a is non-negative; the constraint that c_{a,n_z} is non-negative is
      is sufficient to assure c_a will be as well.

    [1] Poupart, Boutilier. (2003). Bounded Finite State Controllers
    '''

    # Number of states, actions, observations in the POMDP
    nactions, nstates, nobs = pomdp.observation_matrix.shape
    ncontroller = V.shape[0]
    assert V.shape == (ncontroller, nstates)

    T = pomdp.transition_matrix
    O = pomdp.observation_matrix
    R = pomdp.state_action_reward_matrix

    epsilon = cp.Variable(1)
    # We can't have variables with dimensionality greater than 2, so encoding the first axis as a dictionary.
    canz = {
        a: cp.Variable((nobs, ncontroller))
        for a in range(nactions)
    }
    # We define ca as the sum of canz for each action, at an arbitrary observation.
    ca = cp.hstack([cp.sum(canz[a][0]) for a in range(nactions)])

    # Our state constraints are pretty straightforward; We want to find a node using one-step lookahead with a value
    # that dominates our existing node's value for every state.
    reward = R @ ca
    # Summing over controller nodes is left out in Poupart 2003, but is in Grzes, Poupart 2015's
    # Figure 1, which is a description of Poupart 2003. We do it here via this matrix multiply.
    future_reward_given_a_o_ns = [canz[a] @ V for a in range(nactions)]
    ns_o_given_s_a = np.einsum('san,ano->sano', T, O)
    state_constraints = [
        V[node, s] + epsilon <= reward[s] + pomdp.discount_rate * sum([
            # Expected future reward
            cp.sum(cp.multiply(ns_o_given_s_a[s, a], future_reward_given_a_o_ns[a].T))
            for a in range(nactions)
        ])
        for s in range(nstates)
    ]

    prob = cp.Problem(
        cp.Maximize(epsilon),
        [
            # Simplex constraint on c_a.
            cp.sum(ca) == 1
        ] + [
            # Constraint that \sum_{n_z} c_{a,n_z} = c_a for all a, z
            cp.sum(canz[a][o]) == ca[a]
            for a in range(nactions)
            for o in range(nobs)
        ] + [
            # Non-negativity constraint.
            v >= 0 for a, v in canz.items()
        ] + state_constraints)
    prob.solve(solver=solver, verbose=verbose)

    # Unpack our solution.
    action_strategy = ca.value

    observation_strategy = np.stack([
        canz[a].value / action_strategy[a]
        for a in range(nactions)
    ])
    # HACK: for actions with near-0 probabilities, we code in a uniform distribution over next internal states since
    # the above division by a near-0 p(a|s) usually means this doesn't sum to 1 because of numerical errors.
    # We mostly do this because we check that these distributions sum to 1 in other methods.
    observation_strategy[np.isclose(action_strategy, 0), :, :] = 1/ncontroller
    assert np.allclose(observation_strategy.sum(-1), np.ones((nactions, nobs)))

    def add_to_fsc(fsc_action, fsc_state, *, inplace=True):
        if not inplace:
            fsc_action = np.copy(fsc_action)
            fsc_state = np.copy(fsc_state)
        fsc_action[node] = action_strategy
        fsc_state[node] = observation_strategy
        return fsc_action, fsc_state

    return Result(
        epsilon=epsilon.value,
        # HACK: the not isclose check is ensure we're not just a hair above 0
        improved=epsilon.value>0 and not np.isclose(epsilon.value, 0),
        action_strategy=action_strategy,
        observation_strategy=observation_strategy,
        solver_result=Result(problem=prob),
        # We also pull out the tangent belief from the dual of our state constraints.
        tangent_belief=np.concatenate([c.dual_value for c in state_constraints]),
        add_to_fsc=add_to_fsc,
    )


def check_improvement_at_reachable_beliefs(pomdp, beliefs, state_controller_value):
    '''
    This method enumerates reachable beliefs to look for a controller node that could increase value.
    '''
    seen = set()

    for belief in beliefs:
        # Grzes 2015's IPI makes use of an on-policy lookahead; may be worth abstracting this routine for that.
        for ai, a in enumerate(pomdp.action_list):
            for oi, s_ns_o_prob in enumerate(pomdp.predictive_observation_vec(belief, ai)):
                # Not reachable under current belief, so skip.
                if s_ns_o_prob == 0:
                    continue

                # Get resulting belief
                next_belief = pomdp.state_estimator_vec(belief, ai, oi)
                assert np.isclose(np.sum(next_belief), 1)

                # We avoid repeating beliefs
                key = tuple(next_belief.tolist())
                if key in seen:
                    continue
                seen.add(key)

                # Look for improvement at this belief
                r = propose_escape_node(pomdp, next_belief, state_controller_value)

                if r.improved:
                    return r


class FSCBoundedPolicyIteration(Learns):
    def __init__(
        self, *,
        # Number of states the controller should start with.
        controller_state_count,
        iterations=100,
        seed=None,
        convergence_diff=1e-5,
    ):
        self.controller_state_count = controller_state_count
        self.iterations = iterations
        self.seed = seed or np.random.randint(2**30)
        self.convergence_diff = convergence_diff

    def train_on(self, pomdp: TabularPOMDP):
        # Number of states, actions, observations in the POMDP
        nactions, nstates, nobs = pomdp.observation_matrix.shape
        # Number of states in the finite state controller.
        ncontroller = self.controller_state_count

        rng = np.random.default_rng(self.seed)
        def sample_distribution(*size):
            d = rng.uniform(1, 2, size=size)
            return d / d.sum(axis=-1, keepdims=True)
        fsc_action = sample_distribution(ncontroller, nactions)
        fsc_state = sample_distribution(ncontroller, nactions, nobs, ncontroller)

        # HACK: Using torch-based policy evaluation for now.
        def value(fsc_action, fsc_state):
            return stochastic_fsc_policy_evaluation_exact(pomdp, torch.tensor(fsc_action), torch.tensor(fsc_state)).state_controller_value.numpy()

        def assert_value_improvement(V, fsc_fn):
            nextV = value(*fsc_fn())
            assert np.all(np.isclose(nextV, V) | (nextV > V))
            assert np.any(nextV > V)

        V = value(fsc_action, fsc_state)
        converged = False
        for idx in range(self.iterations):
            assert V.shape == (ncontroller, nstates)
            assert ncontroller == fsc_action.shape[0] == fsc_state.shape[0] == fsc_state.shape[-1]
            prev = np.copy(V)
            tangent_beliefs = []
            improved = False

            for n in range(ncontroller):
                r = improve_node(pomdp, V, n)
                if r.improved:
                    improved = True
                    assert_value_improvement(V, lambda: r.add_to_fsc(fsc_action, fsc_state, inplace=False))
                    r.add_to_fsc(fsc_action, fsc_state, inplace=True)
                    V = value(fsc_action, fsc_state)
                else:
                    tangent_beliefs.append(r.tangent_belief)

            if not improved:
                r = check_improvement_at_reachable_beliefs(pomdp, tangent_beliefs, V)
                if r and r.improved:
                    # Can't really assert value improvement here, since we are only
                    # improving value at new nodes.
                    fsc_action, fsc_state = r.add_to_fsc(fsc_action, fsc_state)
                    ncontroller += 1
                    V = value(fsc_action, fsc_state)
                    continue # doing this to skip over the convergence test below.

            if np.abs(prev-V).max() < self.convergence_diff:
                converged = True
                break

        initial_controller_values = V @ pomdp.initial_state_vec
        fsc_initial_state = np.zeros(ncontroller)
        fsc_initial_state[np.argmax(initial_controller_values)] = 1
        return Result(
            converged=converged,
            policy=StochasticFiniteStateController(pomdp, fsc_action, fsc_state, fsc_initial_state),
            value=fsc_initial_state@initial_controller_values,
            state_controller_value=V,
        )