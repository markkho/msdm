import logging
import numpy as np
from abc import abstractmethod
from typing import Set, Sequence, Hashable, Mapping, TypeVar
from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.distributions import FiniteDistribution, DictDistribution

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
            return sorted(states)
        except TypeError: #unsortable
            pass
        return list(states)

    @cached_property
    def state_index(self) -> Mapping[HashableState, int]:
        return {s: i for i, s in enumerate(self.state_list)}

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
            return sorted(actions)
        except TypeError: #unsortable action representation
            pass
        return list(actions)

    @cached_property
    def action_index(self) -> Mapping[HashableAction, int]:
        return {a: i for i, a in enumerate(self.action_list)}

    @cached_property
    def transition_matrix(self) -> np.array:
        ss = self.state_list
        ssi = self.state_index
        aa = self.action_list
        aai = self.action_index
        tf = np.zeros((len(ss), len(aa), len(ss)))
        for s, si in ssi.items():
            # by definition, terminal states lead only to themselves
            # if self.is_terminal(s):
            #     tf[si, :, si] = 1
            #     continue
            for a in self._cached_actions(s):
                for ns, nsp in self._cached_next_state_dist(s, a).items():
                    tf[si, aai[a], ssi[ns]] = nsp
        return tf

    @cached_property
    def action_matrix(self):
        ss = self.state_list
        ssi = self.state_index
        aa = self.action_list
        aai = self.action_index
        am = np.zeros((len(ss), len(aa)))
        for s, si in ssi.items():
            for a in self._cached_actions(s):
                am[si, aai[a]] = 1
        return am

    @cached_property
    def reward_matrix(self):
        ss = self.state_list
        ssi = self.state_index
        aa = self.action_list
        aai = self.action_index
        rf = np.zeros((len(ss), len(aa), len(ss)))
        for s, si in ssi.items():
            # by definition, reward from a terminal state is 0
            # if self.is_terminal(s):
            #     continue
            for a in self._cached_actions(s):
                for ns, p in self._cached_next_state_dist(s, a).items():
                    if p == 0.:
                        continue
                    rf[si, aai[a], ssi[ns]] = self.reward(s, a, ns)
        return rf

    @cached_property
    def state_action_reward_matrix(self):
        rf = self.reward_matrix
        tf = self.transition_matrix
        return np.einsum("san,san->sa", rf, tf)

    @cached_property
    def initial_state_vec(self):
        s0 = self.initial_state_dist()
        return np.array([s0.prob(s) for s in self.state_list])

    @cached_property
    def nonterminal_state_vec(self):
        ss = self.state_list
        return np.array([0 if self.is_terminal(s) else 1 for s in ss])

    @cached_property
    def reachable_state_vec(self):
        reachable = self.reachable_states()
        return np.array([1 if s in reachable else 0 for s in self.state_list])

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
        return np.array([is_absorbing(s) for s in self.state_list])

    @method_cache
    def reachable_states(self) -> Set[HashableState]:
        S0 = self.initial_state_dist().support
        frontier = set(S0)
        visited = set(S0)
        while len(frontier) > 0:
            s = frontier.pop()
            for a in self._cached_actions(s):
                for ns in self._cached_next_state_dist(s, a).support:
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        return visited
