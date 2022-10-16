from typing import Sequence
import numpy as np

from msdm.core.problemclasses.mdp.policy.policy import Policy, PolicyEvaluationResult
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State, Action
from msdm.core.table import Table, ProbabilityTable, TableIndex

class TabularPolicy(ProbabilityTable,Policy):
    def __init__(
        self,
        state_list : Sequence[State],
        action_list : Sequence[Action],
        policy_matrix: np.array
    ):
        ProbabilityTable.__init__(
            self,
            data=policy_matrix,
            table_index=TableIndex(
                field_names=("state", "action"),
                field_domains=(tuple(state_list), tuple(action_list))
            )
        )
        
    def action_dist(self, s):
        return self[s]
    @property
    def state_list(self):
        return self.table_index.field_domains[0]
    @property
    def action_list(self):
        return self.table_index.field_domains[1]
    def __repr__(self):
        return f"{self.__class__.__name__}(" +\
            f"states={self.state_list},\n"+\
            f"\tactions={self.action_list},\n"+\
            f"\tpolicy_matrix={repr(np.array(self))}\n)"

    def evaluate_on(
        self,
        mdp: TabularMarkovDecisionProcess
    ):
        policy_matrix = self.as_matrix(mdp.state_list, mdp.action_list)
        terminal_state_vec = ~mdp.nonterminal_state_vec.astype(bool)
        state_rewards = np.einsum(
            "sa,sa->s",
            policy_matrix, mdp.state_action_reward_matrix
        )
        state_rewards[terminal_state_vec] = 0
        markov_process = np.einsum(
            "san,sa->sn",
            mdp.transition_matrix,
            policy_matrix
        )
        markov_process[terminal_state_vec, :] = 0
        successor_representation = np.linalg.inv(
            np.eye(markov_process.shape[0]) - mdp.discount_rate*markov_process
        )
        
        state_value = np.einsum(
            "sz,z->s",
            successor_representation, state_rewards
        )
        action_value = \
            mdp.state_action_reward_matrix + \
            np.einsum(
                "san,n->sa",
                mdp.discount_rate*mdp.transition_matrix,
                state_value
            )
        state_occupancy = np.einsum(
            "sz,s->z",
            successor_representation,
            mdp.initial_state_vec
        )
        initial_value = state_value.dot(mdp.initial_state_vec)
        state_index = TableIndex(
            field_names=("state", ),
            field_domains=(mdp.state_list, )
        )
        state_action_index = TableIndex(
            field_names=("state", "action", ),
            field_domains=(mdp.state_list, mdp.action_list, )
        )
        return PolicyEvaluationResult(
            state_value=Table(state_value, state_index),
            action_value=Table(action_value, state_action_index),
            initial_value=initial_value,
            state_occupancy=Table(state_occupancy, state_index),
            n_simulations=float('inf')
        )