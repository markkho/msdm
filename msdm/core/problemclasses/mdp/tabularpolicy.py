import numpy as np

from msdm.core.problemclasses.mdp.policy import Policy, PolicyEvaluationResult
from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State, Action
from msdm.core.table import Table, ProbabilityTable, TableIndex
from msdm.core.mdp_tables import StateActionTable, StateTable

class TabularPolicy(StateActionTable,ProbabilityTable,Policy):
    def action_dist(self, s : State) -> DictDistribution:
        return self[s]

    def evaluate_on(
        self,
        mdp: TabularMarkovDecisionProcess
    ):
        assert set(self.action_list) <= set(mdp.action_list), \
            "All policy actions must be in the mdp"
        policy_matrix = np.array(self[mdp.state_list,][:,mdp.action_list])
        absorbing_state_vec = mdp.absorbing_state_vec.astype(bool)
        state_rewards = np.einsum(
            "sa,sa->s",
            policy_matrix, mdp.state_action_reward_matrix
        )
        state_rewards[absorbing_state_vec] = 0
        markov_process = np.einsum(
            "san,sa->sn",
            mdp.transition_matrix,
            policy_matrix
        )
        markov_process[absorbing_state_vec, :] = 0
        successor_representation = np.linalg.inv(
            np.eye(markov_process.shape[0]) - mdp.discount_rate*markov_process
        )
        
        state_value = np.einsum(
            "sz,z->s",
            successor_representation, state_rewards
        )
        action_value = \
            mdp.state_action_reward_matrix + \
            np.log(mdp.action_matrix) + \
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
        return PolicyEvaluationResult(
            state_value=StateTable.from_state_list(
                state_list=mdp.state_list,
                data=state_value
            ),
            action_value=StateActionTable.from_state_action_lists(
                state_list=mdp.state_list,
                action_list=mdp.action_list,
                data=action_value
            ),
            initial_value=initial_value,
            state_occupancy=StateTable.from_state_list(
                state_list=mdp.state_list,
                data=state_occupancy
            ),
            n_simulations=None
        )
