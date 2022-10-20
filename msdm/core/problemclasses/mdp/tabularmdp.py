import logging
import numpy as np
from abc import abstractmethod
from typing import Set, Sequence, Hashable, Mapping, TypeVar
from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.distributions import FiniteDistribution, DictDistribution
from msdm.core.tableindex import domaintuple

logger = logging.getLogger(__name__)

HashableState = TypeVar('HashableState', bound=Hashable)
HashableAction = TypeVar('HashableAction', bound=Hashable)

class TabularMarkovDecisionProcess(MarkovDecisionProcess):
    """
    Tabular MDPs can be fully enumerated (e.g., as matrices) and
    assume states/actions are hashable.
    """
    @classmethod
    def from_matrices(
        cls,
        state_list : Sequence[HashableState],
        action_list : Sequence[HashableAction],
        initial_state_vec : np.array,
        transition_matrix : np.array,
        reward_matrix : np.array,
        nonterminal_state_vec : np.array,
        discount_rate : float,
        action_matrix=None
    ) -> "TabularMarkovDecisionProcess":
        """
        Constructs a Tabular MDP from matrices.
        """
        assert len(state_list) \
            == transition_matrix.shape[0] \
            == transition_matrix.shape[2] \
            == nonterminal_state_vec.shape[0] \
            == initial_state_vec.shape[0] \
            == reward_matrix.shape[0] \
            == reward_matrix.shape[2]
        assert len(action_list) \
            == transition_matrix.shape[1] \
            == reward_matrix.shape[1]
        if action_matrix is not None:
            assert len(action_list) == action_matrix.shape[1]
            assert len(state_list) == action_matrix.shape[0]

        #avoids circular dependency
        from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP

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
            if action_matrix is None:
                return action_list
            return [a for a, ai in enumerate(action_matrix[ss_i[s]]) if ai > 0]
        initial_state_dist = DictDistribution({
            s: p for s, p in zip(state_list, initial_state_vec) if p > 0
        })
        def is_terminal(s):
            return not nonterminal_state_vec[ss_i[s]]
        return QuickTabularMDP(
            next_state_dist=next_state_dist,
            reward=reward,
            actions=actions,
            initial_state_dist=initial_state_dist,
            is_terminal=is_terminal,
            discount_rate=discount_rate
        )

    @abstractmethod
    def next_state_dist(self, s : HashableState, a : HashableAction) -> FiniteDistribution:
        pass

    @abstractmethod
    def initial_state_dist(self) -> FiniteDistribution:
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

    def as_matrices(self):
        return {
            'ss': self.state_list,
            'aa': self.action_list,
            'tf': self.transition_matrix,
            'rf': self.reward_matrix,
            'sarf': self.state_action_reward_matrix,
            's0': self.initial_state_vec,
            'nt': self.nonterminal_state_vec,
            'rs': self.reachable_state_vec,
            'ast': self.absorbing_state_vec
        }

    @cached_property
    def state_list(self) -> Sequence[HashableState]:
        """
        List of states. Note that state ordering is only guaranteed to be
        consistent for a particular TabularMarkovDecisionProcess instance.
        """
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
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = set([])
        for s in self.state_list:
            for a in self.actions(s):
                actions.add(a)
        try:
            return domaintuple(sorted(actions))
        except TypeError: #unsortable action representation
            pass
        return domaintuple(actions)

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
    def nonterminal_state_vec(self):
        nt = np.array([0 if self.is_terminal(s) else 1 for s in self.state_list])
        nt.setflags(write=False)
        return nt

    @cached_property
    def reachable_state_vec(self):
        reachable = self.reachable_states()
        reachable = np.array([1 if s in reachable else 0 for s in self.state_list])
        reachable.setflags(write=False)
        return reachable

    @cached_property
    def absorbing_state_vec(self):
        def is_absorbing(s):
            actions = self.actions(s)
            for a in actions:
                nextstates = self._cached_next_state_dist(s, a).support
                for ns in nextstates:
                    if not self.is_terminal(ns):
                        return False
            return True
        absorbing = np.array([is_absorbing(s) for s in self.state_list])
        absorbing.setflags(write=False)
        return absorbing

    @method_cache
    def reachable_states(self, max_states=float('inf')) -> Set[HashableState]:
        S0 = {e for e, p in self.initial_state_dist().items() if p > 0}
        frontier = set(S0)
        visited = set(S0)
        while max_states > len(frontier) > 0:
            s = frontier.pop()
            for a in self._cached_actions(s):
                for ns, prob in self._cached_next_state_dist(s, a).items():
                    if prob == 0:
                        continue
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        return visited
