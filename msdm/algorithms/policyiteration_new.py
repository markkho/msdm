import warnings
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult

class PolicyIteration(Plans):
    VALUE_DECIMAL_PRECISION = 10
    def __init__(
        self,
        max_iterations=int(1e5),
        _version="vectorized"
    ):
        self.max_iterations = max_iterations
        self._version = _version

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        if self._version == 'vectorized':
            v, q, _, iterations = policy_iteration_vectorized(
                transition_matrix=mdp.transition_matrix,
                terminal_state_vector=~mdp.nonterminal_state_vec.astype(bool),
                discount_rate=mdp.discount_rate,
                reward_matrix=mdp.reward_matrix,
                action_matrix=mdp.action_matrix.astype(bool),
                max_iterations=self.max_iterations
            )
            action_value = {}
            for s in mdp.state_list:
                si = mdp.state_index[s]
                action_value[s] = {}
                for a in mdp.actions(s):
                    ai = mdp.action_index[a]
                    action_value[s][a] = q[si][ai]
            state_value = dict(zip(mdp.state_list, v))
        else:
            raise
        policy = {}
        round_val = lambda v: round(v, self.VALUE_DECIMAL_PRECISION)
        for s in mdp.state_list:
            if len(action_value[s]) == 0:
                continue
            maxq = max([round_val(v) for v in action_value[s].values()])
            max_actions = [a for a in mdp.actions(s) if round_val(action_value[s][a]) == maxq]
            policy[s] = DictDistribution({a: 1/len(max_actions) for a in max_actions})
        return PolicyIterationResult(
            iterations=iterations,
            converged=iterations < (self.max_iterations - 1),
            state_value=state_value,
            initial_value=sum([state_value[s]*p for s, p in mdp.initial_state_dist().items()]),
            action_value=action_value,
            policy=TabularPolicy(policy)
        )

@dataclass
class PolicyIterationResult(PlanningResult):
    iterations : int
    converged : bool
    state_value : dict
    initial_value : float
    action_value: dict
    policy : TabularPolicy

def policy_iteration_vectorized(
    transition_matrix,
    terminal_state_vector,
    discount_rate,
    reward_matrix,
    action_matrix,
    max_iterations=int(1e5),
    value_difference=1e-10,
    policy=None
):
    """
    Implementation of regularized policy iteration with
    an inverse temperature parameter of infinity, which
    yields and equivalent optimal value function.

    This implementation supports arbitrary discounted problems,
    and undiscounted problems where all non-terminal transitions 
    have non-positive rewards.
    """
    assert terminal_state_vector.dtype == bool
    assert action_matrix.dtype == bool
    if discount_rate == 1.0:
        # For undiscounted problems, we only allow "safe actions", which are those
        # that lead to states where it is possible to reach the terminal state w.p. 1
        # TODO: Handling more general case requires doing average-reward optimization
        assert reward_matrix[~terminal_state_vector, :, ~terminal_state_vector].max() <= 0, \
            "Currently, undiscounted problems can only be solved " + \
            "if all non-terminal to non-terminal rewards are negative"
        safe_actions = calculate_safe_actions(
            transition_matrix,
            terminal_state_vector,
            action_matrix,
        )
        safe_action_matrix = np.logical_and(safe_actions, action_matrix.astype(bool))
        action_penalty = np.log(safe_action_matrix)
    else:
        action_penalty = np.log(action_matrix)
    rf_sa = np.einsum("san,san->sa", reward_matrix, transition_matrix)
    rf_sa[terminal_state_vector] = 0
    policy = action_matrix/action_matrix.sum(-1, keepdims=True)
    policy[np.isnan(policy)] = 0

    for i in range(max_iterations):
        mp = np.einsum("san,sa->sn", transition_matrix, policy)
        mp[terminal_state_vector] = 0
        rf_s = np.einsum("sa,sa->s", rf_sa, policy)
        v = np.linalg.solve(np.eye(mp.shape[0]) - discount_rate*mp, rf_s)
        q = rf_sa + (discount_rate*transition_matrix*v[None, None, :]).sum(-1) + action_penalty
        q[terminal_state_vector] = 0
        new_policy = np.isclose(q, np.max(q, axis=-1, keepdims=True), atol=value_difference, rtol=0)
        new_policy = new_policy/new_policy.sum(-1, keepdims=True)
        if np.isclose(new_policy, policy).all():
            break
        policy[:] = new_policy
    policy[~action_matrix] = 0
    v = np.max(q, axis=-1)
    return v, q, policy, i

def calculate_safe_actions(
    transition_matrix,
    terminal_state_vector,
    action_matrix,
):
    """
    Finds safe actions - actions that lead to safe states with probability 1.
    A safe state is one from which a terminal state can be reached
    with probability 1 (or is itself a terminal state).
    """
    # Random walk over transition function, terminating at terminal states,
    # tells us which states are safe
    rand_mp = np.einsum("san,sa->sn", transition_matrix, action_matrix)
    np.divide(rand_mp, rand_mp.sum(-1, keepdims=True), where=(rand_mp > 0), out=rand_mp)
    rand_mp[terminal_state_vector] = 0
    term_visit = np.linalg.solve(np.eye(rand_mp.shape[0]) - rand_mp, terminal_state_vector)
    safe_states = (term_visit > 0) | terminal_state_vector
    
    # safe actions lead to safe states w.p 1
    safe_actions = np.einsum("san,n->sa", transition_matrix, safe_states)
    safe_actions = np.isclose(safe_actions, 1, atol=1e-10, rtol=0)
    return safe_actions