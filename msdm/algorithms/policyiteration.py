from dataclasses import dataclass
from typing import Sequence
import warnings
import numpy as np

from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp.tabularpolicy import TabularPolicy
from msdm.core.algorithmclasses import Plans, PlanningResult
from msdm.core.mdp_tables import StateActionTable, StateTable

class PolicyIteration(Plans):
    def __init__(
        self,
        max_iterations=int(1e5),
        undefined_value=0,
    ):
        self.max_iterations = max_iterations
        self.undefined_value = undefined_value

    def plan_on(self, mdp: TabularMarkovDecisionProcess) -> "PolicyIterationResult":
        return self.batch_plan_on([mdp])[0]
    
    def batch_plan_on(self, mdps: Sequence[TabularMarkovDecisionProcess]) -> Sequence["PolicyIterationResult"]:
        transition_matrices = np.zeros(
            (len(mdps), ) + mdps[0].transition_matrix.shape,
            dtype=mdps[0].transition_matrix.dtype
        )
        discount_rates = np.zeros((len(mdps), ), dtype=type(mdps[0].discount_rate))
        state_action_reward_matrices = np.zeros(
            (len(mdps), ) + mdps[0].transition_matrix.shape[:-1],
            dtype=mdps[0].reward_matrix.dtype
        )
        action_matrices = np.zeros(
            (len(mdps), ) + mdps[0].transition_matrix.shape[:-1],
            dtype=bool
        )
        for mdp_i, mdp in enumerate(mdps):
            if mdp.dead_end_state_vec.any():
                warnings.warn("MDP contains states where no actions can be taken. This can have unanticipated effects.")
            if mdp.recurrent_state_vec.any():
                warnings.warn("MDP contains states that never reach an absorbing state. " +\
                    f"Values for these states will be set using self.undefined_value={self.undefined_value}"
                )
            transition_matrices[mdp_i] = mdp.transition_matrix
            transition_matrix = transition_matrices[mdp_i]
            transition_matrix[mdp.recurrent_state_vec,] = 0
            transition_matrix[mdp.absorbing_state_vec,] = 0
            discount_rates[mdp_i] = mdp.discount_rate
            state_action_reward_matrix = np.einsum("san,san->sa", transition_matrix, mdp.reward_matrix)
            state_action_reward_matrices[mdp_i] = state_action_reward_matrix
            action_matrices[mdp_i] = mdp.action_matrix.astype(bool)
        
        policy_norm = action_matrices.sum(-1, keepdims=True)
        policy_norm[policy_norm == 0] = 1
        policy_matrices = action_matrices/policy_norm

        state_values_batch, action_values_batch, policy_matrix_batch, iterations = policy_iteration_vectorized(
            transition_matrix=transition_matrices,
            discount_rate=discount_rates,
            state_action_reward_matrix=state_action_reward_matrices,
            action_matrix=action_matrices,
            max_iterations=self.max_iterations,
            policy_matrix=policy_matrices,
        )

        results = []
        for mdp, state_values, action_values, policy_matrix in zip(
            mdps, state_values_batch, action_values_batch, policy_matrix_batch
        ):
            state_values[mdp.recurrent_state_vec,] = self.undefined_value
            action_values[mdp.recurrent_state_vec,] = self.undefined_value
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
            results.append(PolicyIterationResult(
                iterations=iterations,
                state_value=state_values,
                action_value=action_values,
                converged=iterations < (self.max_iterations - 1),
                initial_value=sum([state_values[s]*p for s, p in mdp.initial_state_dist().items()]),
                policy=policy
            ))
        return results

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
    discount_rate,
    state_action_reward_matrix,
    action_matrix,
    policy_matrix,
    max_iterations=int(1e5),
    value_difference=1e-10,
):
    """
    Implementation of regularized policy iteration with
    an inverse temperature parameter of infinity, which
    yields an equivalent optimal value function.
    
    Note that the first dimension of all inputs is a batch dimension.
    """
    assert action_matrix.dtype == bool
    action_penalty = np.log(action_matrix)
    n_states = transition_matrix.shape[1]
    eye = np.eye(n_states)
    
    for i in range(max_iterations):
        mp = np.einsum("bsan,bsa,b->bsn", transition_matrix, policy_matrix, discount_rate)
        rf_s = np.einsum("bsa,bsa->bs", state_action_reward_matrix, policy_matrix)
        v = np.linalg.solve(eye[None,...] - mp, rf_s)
        next_q = np.einsum("b,bsan,bn->bsa", discount_rate, transition_matrix, v)
        q = state_action_reward_matrix + action_penalty + next_q 
        new_policy = np.isclose(q, np.max(q, axis=-1, keepdims=True), atol=value_difference, rtol=0)
        new_policy = new_policy/new_policy.sum(-1, keepdims=True)
        if np.isclose(new_policy, policy_matrix, atol=value_difference, rtol=0).all():
            break
        policy_matrix = new_policy
    policy_matrix[~action_matrix] = 0
    v = np.max(q, axis=-1)
    return v, q, policy_matrix, i
