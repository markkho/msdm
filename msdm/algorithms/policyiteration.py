from dataclasses import dataclass
import warnings
import numpy as np

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp.tabularpolicy import TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.mdp_tables import StateActionTable, StateTable

class PolicyIteration(Plans):
    VALUE_DECIMAL_PRECISION = 10
    def __init__(
        self,
        max_iterations=int(1e5),
        undefined_value=0,
    ):
        self.max_iterations = max_iterations
        self.undefined_value = undefined_value

    def plan_on(self, mdp: TabularMarkovDecisionProcess):
        if mdp.dead_end_state_vec.any():
            warnings.warn("MDP contains states where no actions can be taken. This can have unanticipated effects.")
        transition_matrix = mdp.transition_matrix.copy()
        transition_matrix[mdp.recurrent_state_vec,] = 0
        state_values, action_values, policy_matrix, iterations = policy_iteration_vectorized(
            transition_matrix=transition_matrix,
            terminal_state_vector=~mdp.transient_state_vec.astype(bool),
            discount_rate=mdp.discount_rate,
            reward_matrix=mdp.reward_matrix,
            action_matrix=mdp.action_matrix.astype(bool),
            max_iterations=self.max_iterations
        )
        state_values[mdp.recurrent_state_vec,] = self.undefined_value
        action_values[mdp.recurrent_state_vec,] = self.undefined_value
            
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
