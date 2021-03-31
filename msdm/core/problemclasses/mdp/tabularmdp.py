import logging
import numpy as np
from abc import abstractmethod
from msdm.core.problemclasses.mdp import MarkovDecisionProcess
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.distributions import DiscreteDistribution

logger = logging.getLogger(__name__)

class TabularMarkovDecisionProcess(MarkovDecisionProcess):
    @abstractmethod
    def next_state_dist(self, s, a) -> DiscreteDistribution:
        pass

    @method_cache
    def _cached_next_state_dist(self, s, a) -> DiscreteDistribution:
        '''
        We prefer using this cached version of next_state_dist when possible.
        '''
        return self.next_state_dist(s, a)

    """Tabular MDPs can be fully enumerated (e.g., as matrices)"""
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
    def state_list(self):
        """
        List of states. Note that state ordering is only guaranteed to be
        consistent for a particular TabularMarkovDecisionProcess instance.
        """
        logger.info("State space unspecified; performing reachability analysis.")
        return list(self.reachable_states())

    @cached_property
    def action_list(self):
        """
        List of actions. Note that action ordering is only guaranteed to be
        consistent for a particular TabularMarkovDecisionProcess instance.
        """
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = set([])
        for s in self.state_list:
            for a in self.actions(s):
                actions.add(a)
        return list(actions)

    @cached_property
    def transition_matrix(self):
        ss = self.state_list
        aa = self.action_list
        tf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            state_actions = self.actions(s)
            for ai, a in enumerate(aa):
                if a not in state_actions:
                    continue
                nsdist = self._cached_next_state_dist(s, a)
                for ns, nsp in nsdist.items():
                    tf[si, ai, ss.index(ns)] = nsp
        return tf

    @cached_property
    def action_matrix(self):
        ss = self.state_list
        aa = self.action_list
        am = np.zeros((len(ss), len(aa)))
        for (si, ai), _ in np.ndenumerate(am):
            s, a = ss[si], aa[ai]
            if a in self.actions(s):
                p = 1
            else:
                p = 0
            am[si, ai] = p
        return am


    @cached_property
    def reward_matrix(self):
        ss = self.state_list
        aa = self.action_list
        rf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            state_actions = self.actions(s)
            for ai, a in enumerate(aa):
                if a not in state_actions:
                    continue
                nsdist = self._cached_next_state_dist(s, a)
                for ns in nsdist.support:
                    nsi = ss.index(ns)
                    rf[si, ai, nsi] = self.reward(s, a, ns)
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
    def reachable_states(self):
        S0 = self.initial_state_dist().support
        frontier = set(S0)
        visited = set(S0)
        while len(frontier) > 0:
            s = frontier.pop()
            for a in self.actions(s):
                for ns in self._cached_next_state_dist(s, a).support:
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        return visited
