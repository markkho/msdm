import json, logging
import numpy as np
from msdm.core.problemclasses.mdp import MarkovDecisionProcess

from msdm.core.assignment.assignmentset import AssignmentSet as Set
logger = logging.getLogger(__name__)

class TabularMarkovDecisionProcess(MarkovDecisionProcess):
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

    @property
    def state_list(self):
        try:
            return self._states
        except AttributeError:
            pass
        logger.info("State space unspecified; performing reachability analysis.")
        self._states = \
            sorted(self.reachable_states(),
                key=self.hash_state
            )
        return self._states

    @property
    def action_list(self):
        try:
            return self._actions
        except AttributeError:
            pass
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = Set([])
        for s in self.state_list:
            for a in self.actions(s):
                actions.add(a)
        self._actions = sorted(actions, 
                key=self.hash_action
            )
        return self._actions

    @property
    def transition_matrix(self):
        try:
            return self._tfmatrix
        except AttributeError:
            pass
        ss = self.state_list
        aa = self.action_list
        tf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            for ai, a in enumerate(aa):
                nsdist = self.next_state_dist(s, a)
                for ns, nsp in nsdist.items(probs=True):
                    tf[si, ai, ss.index(ns)] = nsp
        self._tfmatrix = tf
        return self._tfmatrix

    @property
    def action_matrix(self):
        try:
            return self._actmatrix
        except AttributeError:
            pass
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
        self._actmatrix = am
        return self._actmatrix


    @property
    def reward_matrix(self):
        try:
            return self._rfmatrix
        except AttributeError:
            pass
        ss = self.state_list
        aa = self.action_list
        rf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            for ai, a in enumerate(aa):
                nsdist = self.next_state_dist(s, a)
                for ns in nsdist.support:
                    nsi = ss.index(ns)
                    rf[si, ai, nsi] = self.reward(s, a, ns)
        self._rfmatrix = rf
        return self._rfmatrix

    @property
    def state_action_reward_matrix(self):
        try:
            return self._sarfmatrix
        except AttributeError:
            pass
        rf = self.reward_matrix
        tf = self.transition_matrix
        self._sarfmatrix = np.einsum("san,san->sa", rf, tf)
        return self._sarfmatrix

    @property
    def initial_state_vec(self):
        try:
            return self._s0vec
        except AttributeError:
            pass
        s0 = self.initial_state_dist()
        self._s0vec = np.array([s0.prob(s) for s in self.state_list])
        return self._s0vec

    @property
    def nonterminal_state_vec(self):
        try:
            return self._ntvec
        except AttributeError:
            pass
        ss = self.state_list
        self._ntvec = np.array([0 if self.is_terminal(s) else 1 for s in ss])
        return self._ntvec

    @property
    def reachable_state_vec(self):
        try:
            return self._reachablevec
        except AttributeError:
            pass
        reachable = self.reachable_states()
        self._reachablevec = np.array \
            ([1 if s in reachable else 0 for s in self.state_list])
        return self._reachablevec

    @property
    def absorbing_state_vec(self):
        try:
            return self._absorbingstatevec
        except AttributeError:
            pass
        def is_absorbing(s):
            actions = self.actions(s)
            for a in actions:
                nextstates = self.next_state_dist(s, a).support
                for ns in nextstates:
                    if not self.is_terminal(ns):
                        return False
            return True
        self._absorbingstatevec = np.array([is_absorbing(s) for s in self.state_list])
        return self._absorbingstatevec

    def reachable_states(self):
        S0 = self.initial_state_dist().support
        frontier = Set(S0)
        visited = Set(S0)
        while len(frontier) > 0:
            s = frontier.pop()
            for a in self.actions(s):
                for ns in self.next_state_dist(s, a).support:
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        return visited
