from typing import Sequence
import numpy as np

from msdm.core.problemclasses.mdp.policy.policy import Policy, PolicyEvaluationResult
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, State, Action
from msdm.core.table import Table

class TabularPolicy(Table,Policy):
    def __init__(
        self,
        states : Sequence[State],
        actions : Sequence[Action],
        policy_matrix: np.array
    ):
        assert policy_matrix.shape == (len(states), len(actions))
        super().__init__(policy_matrix, (states, actions), probabilities=True)
        self._states = tuple(states)
        self._actions = tuple(actions)
        
    def action_dist(self, s):
        return self[s]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(\n\tstates={self._states}, \n\tactions={self._actions}, \n\tpolicy_matrix={repr(self._values)}\n)"
    
    def as_matrix(self, states, actions):
        if tuple(states) == self._states and tuple(actions) == self._actions:
            return self._values
        policy_matrix = np.zeros((len(states), len(actions)))
        for si, s in enumerate(states):
            self_si = self._state_index[s]
            for ai, a in enumerate(actions):
                self_ai = self._action_index[a]
                policy_matrix[si, ai] = self._values[self_si, self_ai]
        policy_matrix.setflags(write=False)
        return policy_matrix

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
        return PolicyEvaluationResult(
            state_value=Table(state_value, (mdp.state_list,)),
            action_value=Table(action_value, (mdp.state_list, mdp.action_list)),
            initial_value=initial_value,
            state_occupancy=Table(state_occupancy, (mdp.state_list,)),
            n_simulations=float('inf')
        )