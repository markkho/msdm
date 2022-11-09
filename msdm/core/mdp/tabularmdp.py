import logging
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from abc import abstractmethod
from typing import Set, Sequence, Hashable, Mapping, TypeVar
from msdm.core.mdp.mdp import MarkovDecisionProcess
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.distributions import FiniteDistribution, DictDistribution
from msdm.core.table import domaintuple

logger = logging.getLogger(__name__)

HashableState = TypeVar('HashableState', bound=Hashable)
HashableAction = TypeVar('HashableAction', bound=Hashable)

class TabularMarkovDecisionProcess(MarkovDecisionProcess):
    """
    Tabular MDPs can be fully enumerated (e.g., as matrices) and
    assume states/actions are hashable.
    """

    ########################################
    #              Constructors            #
    ########################################
    @classmethod
    def from_matrices(
        cls,
        state_list : Sequence[HashableState],
        action_list : Sequence[HashableAction],
        initial_state_vec : np.array,
        transition_matrix : np.array,
        action_matrix : np.array,
        reward_matrix : np.array,
        absorbing_state_vec : np.array,
        discount_rate : float,
    ) -> "TabularMarkovDecisionProcess":
        assert len(state_list) \
            == transition_matrix.shape[0] \
            == transition_matrix.shape[2] \
            == action_matrix.shape[0] \
            == initial_state_vec.shape[0] \
            == absorbing_state_vec.shape[0] \
            == reward_matrix.shape[0] \
            == reward_matrix.shape[2] 
        assert len(action_list) \
            == transition_matrix.shape[1] \
            == action_matrix.shape[1] \
            == reward_matrix.shape[1]

        #avoids circular dependency
        from msdm.core.mdp.quickmdp import QuickTabularMDP

        ss_i = {s: i for i, s in enumerate(state_list)} #state indices
        aa_i = {a: i for i, a in enumerate(action_list)}
        def next_state_dist(s, a):
            probs = transition_matrix[ss_i[s], aa_i[a], :]
            return DictDistribution({
                ns: p for ns, p in zip(state_list, probs) if p > 0
            })
        def reward(s, a, ns):
            return reward_matrix[ss_i[s], aa_i[a], ss_i[ns]]
        def actions(s):
            available_actions = action_matrix[state_list.index(s)]
            available_actions = [a for a, aa in zip(action_list, available_actions) if aa]
            return tuple(available_actions)
        initial_state_dist = DictDistribution({
            s: p for s, p in zip(state_list, initial_state_vec) if p > 0
        })
        def is_absorbing(s):
            return absorbing_state_vec[ss_i[s]]
        mdp = QuickTabularMDP(
            next_state_dist=next_state_dist,
            reward=reward,
            actions=actions,
            initial_state_dist=initial_state_dist,
            is_absorbing=is_absorbing,
            discount_rate=discount_rate
        )
        mdp._state_list = domaintuple(state_list)
        mdp._action_list = domaintuple(action_list)
        return mdp


    ########################################
    #         Functional interface         #
    ########################################
    @abstractmethod
    def initial_state_dist(self) -> FiniteDistribution:
        pass

    @abstractmethod
    def next_state_dist(self, s : HashableState, a : HashableAction) -> FiniteDistribution:
        pass

    @method_cache
    def _cached_next_state_dist(self, s : HashableState, a : HashableAction) -> FiniteDistribution:
        '''
        We prefer using this cached version of next_state_dist when possible.
        '''
        return self.next_state_dist(s, a)

    @method_cache
    def _cached_actions(self, s : HashableState) -> Sequence[HashableAction]:
        return self.actions(s)

    @cached_property
    def state_list(self) -> Sequence[HashableState]:
        """
        List of states. Note that state ordering is only guaranteed to be
        consistent for a particular TabularMarkovDecisionProcess instance.
        """
        try:
            return domaintuple(self._state_list)
        except AttributeError:
            pass
        logger.info("State space unspecified; performing reachability analysis.")
        states = self.reachable_states()
        try:
            return domaintuple(sorted(states))
        except TypeError: #unsortable
            pass
        return domaintuple(states)

    @cached_property
    def action_list(self) -> Sequence[HashableAction]:
        """
        List of actions. Note that action ordering is only guaranteed to be
        consistent for a particular TabularMarkovDecisionProcess instance.
        """
        try:
            return domaintuple(self._action_list)
        except AttributeError:
            pass
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = set([])
        for s in self.state_list:
            for a in self._cached_actions(s):
                actions.add(a)
        try:
            return domaintuple(sorted(actions))
        except TypeError: #unsortable action representation
            pass
        return domaintuple(actions)


    ########################################
    #             Matrix interface         #
    ########################################
    @cached_property
    def transition_matrix(self) -> np.array:
        tf = np.zeros((
            len(self.state_list),
            len(self.action_list), 
            len(self.state_list)
        ))
        for si, s in enumerate(self.state_list):
            for a in self._cached_actions(s):
                ai = self.action_list.index(a)
                for ns, nsp in self._cached_next_state_dist(s, a).items():
                    nsi = self.state_list.index(ns)
                    tf[si, ai, nsi] = nsp
        tf.setflags(write=False)
        return tf

    @cached_property
    def action_matrix(self):
        am = np.zeros((
            len(self.state_list),
            len(self.action_list), 
        ))
        for si, s in enumerate(self.state_list):
            for a in self._cached_actions(s):
                ai = self.action_list.index(a)
                am[si, ai] = 1
        am.setflags(write=False)
        return am

    @cached_property
    def reward_matrix(self):
        rf = np.zeros((
            len(self.state_list),
            len(self.action_list), 
            len(self.state_list)
        ))
        for si, s in enumerate(self.state_list):
            for a in self._cached_actions(s):
                ai = self.action_list.index(a)
                for ns, p in self._cached_next_state_dist(s, a).items():
                    nsi = self.state_list.index(ns)
                    if p == 0.:
                        continue
                    rf[si, ai, nsi] = self.reward(s, a, ns)
        rf.setflags(write=False)
        return rf

    @cached_property
    def state_action_reward_matrix(self):
        rf = self.reward_matrix
        tf = self.transition_matrix
        sa_rf = np.einsum("san,san->sa", rf, tf)
        sa_rf.setflags(write=False)
        return sa_rf

    @cached_property
    def initial_state_vec(self):
        s0 = self.initial_state_dist()
        s0 = np.array([s0.prob(s) for s in self.state_list])
        s0.setflags(write=False)
        return s0
    
    @cached_property
    def absorbing_state_vec(self) -> np.ndarray:
        """
        Absorbing states are states that only have actions that self-loop and return 
        a reward of 0, or are explicitly marked as absorbing with `is_absorbing`.
        """
        self_looping = np.diagonal(self.transition_matrix, axis1=0, axis2=2).T
        self_looping = (self_looping == 1) | ~self.action_matrix.astype(bool)
        self_looping = self_looping.all(-1)
        self_looping &= ~self.dead_end_state_vec
        zero_reward = (self.reward_matrix == 0).all(axis=(1, 2))
        absorbing_state_vec = np.array([self.is_absorbing(s) for s in self.state_list], dtype=bool)
        absorbing_state_vec = (self_looping & zero_reward) | absorbing_state_vec
        absorbing_state_vec.setflags(write=False)
        return absorbing_state_vec
    
    @cached_property
    def _unable_to_reach_absorbing(self) -> np.ndarray:
        """
        States that can never access an absorbing state under any policy.
        """
        if self.discount_rate < 1.0:
            unable_to_reach_absorbing = np.zeros(len(self.state_list), dtype=bool)
            unable_to_reach_absorbing.setflags(write=False)
            return unable_to_reach_absorbing
        valid_transitions = \
            (self.transition_matrix > 0) & self.action_matrix.astype(bool)[:, :, None]
        adjacency = valid_transitions.any(axis=1)
        accessibilty = floyd_warshall(adjacency)
        unable_to_reach_absorbing = (accessibilty[:, self.absorbing_state_vec] == float('inf')).all(-1)
        unable_to_reach_absorbing.setflags(write=False)
        return unable_to_reach_absorbing
    
    @cached_property
    def dead_end_state_vec(self):
        dead_ends = (~self.action_matrix.astype(bool)).all(-1)
        dead_ends.setflags(write=False)
        return dead_ends

    @cached_property
    def reachable_state_vec(self):
        reachable = self.reachable_states()
        reachable = np.array([1 if s in reachable else 0 for s in self.state_list], dtype=bool)
        reachable.setflags(write=False)
        return reachable

    def as_matrices(self):
        return {
            'ss': self.state_list,
            'aa': self.action_list,
            'tf': self.transition_matrix,
            'rf': self.reward_matrix,
            'sarf': self.state_action_reward_matrix,
            's0': self.initial_state_vec,
            'rs': self.reachable_state_vec,
        }
