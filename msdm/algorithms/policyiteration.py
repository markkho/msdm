from dataclasses import dataclass
from typing import Sequence
import warnings
import numpy as np

from msdm.core.mdp import TabularMarkovDecisionProcess, TabularPolicy, \
    StateActionTable, StateTable
from msdm.core.algorithmclasses import Plans, PlanningResult

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
            if mdp._unable_to_reach_absorbing.any():
                warnings.warn("MDP contains states that never reach an absorbing state. " +\
                    f"Values for these states will be set using self.undefined_value={self.undefined_value}"
                )
            transition_matrices[mdp_i] = mdp.transition_matrix
            transition_matrix = transition_matrices[mdp_i]
            transition_matrix[mdp._unable_to_reach_absorbing,] = 0
            transition_matrix[mdp.absorbing_state_vec,] = 0
            discount_rates[mdp_i] = mdp.discount_rate
            state_action_reward_matrix = np.einsum("san,san->sa", transition_matrix, mdp.reward_matrix)
            state_action_reward_matrices[mdp_i] = state_action_reward_matrix
            action_matrices[mdp_i] = mdp.action_matrix.astype(bool)
        
        policy_norm = action_matrices.sum(-1, keepdims=True)
        policy_norm[policy_norm == 0] = 1 #this is to handle deadends
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
            state_values[mdp._unable_to_reach_absorbing,] = self.undefined_value
            action_values[mdp._unable_to_reach_absorbing,] = self.undefined_value
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

from msdm.core.table.tablemisc import dataclass_repr_html_MixIn
@dataclass
class PolicyIterationResult(PlanningResult,dataclass_repr_html_MixIn):
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
):
    """
    Implementation of regularized policy iteration with
    an inverse temperature parameter of infinity, which
    yields an equivalent optimal value function.
    
    Note that the first dimension of all inputs is a batch dimension.
    """
    assert action_matrix.dtype == bool

    nbatches, nstates, nactions, _ = transition_matrix.shape
    eye = np.eye(nstates)
    cumulant_matrix = np.zeros((nbatches, nstates, nstates))
    state_rewards = np.zeros((nbatches, nstates))
    state_values = np.zeros((nbatches, nstates))
    action_values = np.zeros((nbatches, nstates, nactions))
    
    for i in range(max_iterations):
        cumulant_matrix = np.einsum(
            "bsan,bsa,b->bsn",
            transition_matrix,
            policy_matrix,
            discount_rate,
            out=cumulant_matrix
        )
        cumulant_matrix[:] = eye[np.newaxis, ...] - cumulant_matrix
        state_rewards = np.einsum(
            "bsa,bsa->bs",
            state_action_reward_matrix,
            policy_matrix,
            out=state_rewards
        )
        state_values = \
            np.linalg.solve(
                cumulant_matrix,
                state_rewards,
            )
        action_values = np.einsum(
            "b,bsan,bn->bsa",
            discount_rate,
            transition_matrix,
            state_values,
            out=action_values
        )
        action_values[:] = state_action_reward_matrix + action_values 
        action_values[~action_matrix] = float('-inf')
        new_policy = np.isclose(
            action_values, np.max(action_values, axis=-1, keepdims=True),
        )
        new_policy = new_policy/new_policy.sum(-1, keepdims=True)
        if np.isclose(new_policy, policy_matrix).all():
            break
        policy_matrix = new_policy
    state_values = np.max(action_values, axis=-1)
    return state_values, action_values, policy_matrix, i
