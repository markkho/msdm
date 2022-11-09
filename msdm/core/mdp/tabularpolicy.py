import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import warnings

from msdm.core.distributions import DictDistribution
from msdm.core.mdp.mdp import State
from msdm.core.mdp.policy import Policy, PolicyEvaluationResult
from msdm.core.mdp.tabularmdp import TabularMarkovDecisionProcess
from msdm.core.mdp.tables import StateActionTable, StateTable
from msdm.core.table import ProbabilityTable

class TabularPolicy(StateActionTable,ProbabilityTable,Policy):
    def action_dist(self, s : State) -> DictDistribution:
        return self[s]

    def evaluate_on(
        self,
        mdp: TabularMarkovDecisionProcess,
    ):
        assert set(self.action_list) <= set(mdp.action_list), \
            "All policy actions must be in the mdp"
        if mdp.discount_rate < 1.0:
            return self._evaluate_on_discounted(mdp)
        elif mdp.discount_rate == 1.0:
            return self._evaluate_on_undiscounted(
                mdp=mdp,
                undiscounted_reward_criteria='total_reward'
            )
        else:
            raise ValueError("Discount rate must be in >= 0 and <= 1")

    def _evaluate_on_discounted(
        self, 
        mdp: TabularMarkovDecisionProcess,
    ):
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
    
    def _evaluate_on_undiscounted(
        self,
        mdp: TabularMarkovDecisionProcess,
        undiscounted_reward_criteria="total_reward"
    ):
        assert (mdp.state_action_reward_matrix <= 0).all(), \
            "Only negative rewards are currently supported for undiscounted evaluation"
        assert undiscounted_reward_criteria in ["total_reward"], \
            "Only total reward is currently supported for undiscounted evaluation"
        policy_matrix = np.array(self[mdp.state_list,][:,mdp.action_list])
        absorbing_state_vec = mdp.absorbing_state_vec.astype(bool)
        state_rewards = np.einsum(
            "sa,sa->s",
            policy_matrix, mdp.state_action_reward_matrix
        )
        state_rewards[absorbing_state_vec] = 0

        # Construct a markov chain and analyze it to get the
        # (non-absorbing) recurrent states: states that never
        # access an absorbing state
        markov_process = np.einsum(
            "san,sa->sn",
            mdp.transition_matrix,
            policy_matrix
        )
        markov_process[absorbing_state_vec, :] = 0
        accessible = floyd_warshall(markov_process > 0) < float('inf')
    
        # State i is transient if we might never return to it.
        # That is, it can access a state from which
        # it cannot return or if its outgoing probability is less than 1
        # states that are not transient are recurrent
        transient = (accessible & ~accessible.T).any(-1)
        transient = (transient | (markov_process.sum(-1) < 1)) & ~absorbing_state_vec
        recurrent_states = ~transient & ~absorbing_state_vec
        negative_recurrent_states = recurrent_states & (state_rewards < 0)
        negative_recurrent_accessible_states = accessible[:, negative_recurrent_states].any(-1)
        initial_accessible_recurrent_states = \
            accessible[mdp.initial_state_vec > 0].any(0) & recurrent_states
        
        markov_process[recurrent_states] = 0
        successor_representation = np.linalg.inv(
            np.eye(markov_process.shape[0]) - markov_process
        )
        
        state_value = np.einsum(
            "sz,z->s",
            successor_representation, state_rewards
        )
        state_value[negative_recurrent_accessible_states] = float('-inf')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            future_action_value = mdp.transition_matrix*state_value[None, None, :]
            future_action_value[np.isnan(future_action_value)] = 0
            future_action_value = future_action_value.sum(-1)

        action_value = \
            mdp.state_action_reward_matrix + \
            np.log(mdp.action_matrix) + \
            future_action_value
        state_occupancy = np.einsum(
            "sz,s->z",
            successor_representation,
            mdp.initial_state_vec
        )
        state_occupancy[initial_accessible_recurrent_states] = float('inf')
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            initial_value = state_value*mdp.initial_state_vec
            initial_value[np.isnan(initial_value)] = 0
            initial_value = initial_value.sum()
            
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