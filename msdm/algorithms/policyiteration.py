from dataclasses import dataclass
import numpy as np

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp.policy.tabularpolicy import TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.mdp_tables import StateActionTable, StateTable

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
            state_values, action_values, policy_matrix, iterations = policy_iteration_vectorized(
                transition_matrix=mdp.transition_matrix,
                terminal_state_vector=~mdp.transient_state_vec.astype(bool),
                discount_rate=mdp.discount_rate,
                reward_matrix=mdp.reward_matrix,
                action_matrix=mdp.action_matrix.astype(bool),
                max_iterations=self.max_iterations
            )
            # terminal dead ends are handled by turning them into a uniform distribution
            # since they are all equally -inf, but non-terminal dead end states 
            # are allowed to have their single action taken
            policy_matrix = np.isclose(
                action_values,
                np.max(action_values, axis=-1, keepdims=True),
                atol=10**(-self.VALUE_DECIMAL_PRECISION),
                rtol=0
            )
            policy_matrix = policy_matrix/policy_matrix.sum(-1, keepdims=True)
            single_action_states = mdp.action_matrix.sum(-1) == 1
            policy_matrix[single_action_states] = mdp.action_matrix[single_action_states]
            policy=TabularPolicy.from_state_action_lists(
                state_list=mdp.state_list,
                action_list=mdp.action_list,
                data=policy_matrix
            )
            state_values=StateTable.from_state_list(
                state_list=mdp.state_list,
                data=state_values
            )
            action_values=StateActionTable.from_state_action_lists(
                state_list=mdp.state_list,
                action_list=mdp.action_list,
                data=action_values
            )
        else:
            raise ValueError
        return PolicyIterationResult(
            iterations=iterations,
            state_value=state_values,
            action_value=action_values,
            converged=iterations < (self.max_iterations - 1),
            initial_value=sum([state_values[s]*p for s, p in mdp.initial_state_dist().items()]),
            policy=policy
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
    policy_norm = action_matrix.sum(-1, keepdims=True)
    policy_norm[policy_norm == 0] = 1
    policy = action_matrix/policy_norm

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