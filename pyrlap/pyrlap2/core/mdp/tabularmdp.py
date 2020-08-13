from itertools import product
import warnings, json, logging
import numpy as np
from pyrlap.pyrlap2.core.mdp import ANDMarkovDecisionProcess, \
    MarkovDecisionProcess

from pyrlap.pyrlap2.core.assignmentmap import AssignmentMap as Dict
from pyrlap.pyrlap2.core.assignmentset import AssignmentSet as Set
logger = logging.getLogger(__name__)

class TabularMarkovDecisionProcess(MarkovDecisionProcess):
    """Tabular MDPs can be fully enumerated (e.g., as matrices)"""
    def asMatrices(self):
        return {
            'ss': self.states,
            'aa': self.actions,
            'tf': self.transitionmatrix,
            'rf': self.rewardmatrix,
            'sarf': self.stateactionrewardmatrix,
            's0': self.initialstatevec,
            'nt': self.nonterminalstatevec,
            'rs': self.reachablestatevec,
            'ast': self.absorbingstatevec
        }

    @property
    def states(self):
        try:
            return self._states
        except AttributeError:
            pass
        logger.info("State space unspecified; performing reachability analysis.")
        self._states = \
            sorted(self.getReachableStates(), 
                key=lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else d
            )
        return self._states

    @property
    def actions(self):
        try:
            return self._actions
        except AttributeError:
            pass
        logger.info("Action space unspecified; performing reachability analysis.")
        actions = Set([])
        for s in self.states:
            for a in self.getActionDist(s).support:
                actions.add(a)
        self._actions = sorted(actions, 
                key=lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else d
            )
        return self._actions

    @property
    def transitionmatrix(self):
        try:
            return self._tfmatrix
        except AttributeError:
            pass
        ss = self.states
        aa = self.actions
        tf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            for ai, a in enumerate(aa):
                nsdist = self.getNextStateDist(s, a)
                for nsi, ns in enumerate(ss):
                    tf[si, ai, nsi] = nsdist.prob(ns)
        self._tfmatrix = tf
        return self._tfmatrix

    @property
    def actionmatrix(self):
        try:
            return self._actmatrix
        except AttributeError:
            pass
        ss = self.states
        aa = self.actions
        am = np.zeros((len(ss), len(aa)))
        for (si, ai), _ in np.ndenumerate(am):
            s, a = ss[si], aa[ai]
            p = self.getActionDist(s).prob(a)
            am[si, ai] = p
        self._actmatrix = am
        return self._actmatrix


    @property
    def rewardmatrix(self):
        try:
            return self._rfmatrix
        except AttributeError:
            pass
        ss = self.states
        aa = self.actions
        rf = np.zeros((len(ss), len(aa), len(ss)))
        for si, s in enumerate(ss):
            for ai, a in enumerate(aa):
                nsdist = self.getNextStateDist(s, a)
                for nsi, ns in enumerate(ss):
                    if ns not in nsdist.support:
                        continue
                    r = self.getReward(s, a, ns)
                    rf[si, ai, nsi] = r
        self._rfmatrix = rf
        return self._rfmatrix

    @property
    def stateactionrewardmatrix(self):
        try:
            return self._sarfmatrix
        except AttributeError:
            pass
        rf = self.rewardmatrix
        tf = self.transitionmatrix
        self._sarfmatrix = np.einsum("san,san->sa", rf, tf)
        return self._sarfmatrix

    @property
    def initialstatevec(self):
        try:
            return self._s0vec
        except AttributeError:
            pass
        s0 = self.getInitialStateDist()
        return np.array([s0.prob(s) for s in self.states])

    @property
    def nonterminalstatevec(self):
        try:
            return self._ntvec
        except AttributeError:
            pass
        ss = self.states
        self._ntvec = np.array([0 if self.isTerminal(s) else 1 for s in ss])
        return self._ntvec

    @property
    def reachablestatevec(self):
        try:
            return self._reachablevec
        except AttributeError:
            pass
        reachable = self.getReachableStates()
        self._reachablevec = np.array \
            ([1 if s in reachable else 0 for s in self.states])
        return self._reachablevec

    @property
    def absorbingstatevec(self):
        try:
            return self._absorbingstatevec
        except AttributeError:
            pass
        def isAbsorbing(s):
            actions = self.getActionDist(s).support
            for a in actions:
                nextstates = self.getNextStateDist(s, a).support
                for ns in nextstates:
                    if not self.isTerminal(ns):
                        return False
            return True
        self._absorbingstatevec = np.array([isAbsorbing(s) for s in self.states])
        return self._absorbingstatevec

    

    def getReachableStates(self):
        S0 = self.getInitialStateDist().support
        frontier = Set(S0)
        visited = Set(S0)
        while len(frontier) > 0:
            s = frontier.pop()
            for a in self.getActionDist(s).support:
                for ns in self.getNextStateDist(s, a).support:
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        return visited

    def __and__(self, other: "TabularMarkovDecisionProcess"):
        assert isinstance(other, TabularMarkovDecisionProcess)
        assert all(s == z for s, z in zip(self.states, other.states)), \
            "State spaces not aligned"
        assert all(a == b for a, b in zip(self.actions, other.actions)), \
            "Action spaces not aligned"
        return ANDTabularMarkovDecisionProcess(self, other)

class ANDTabularMarkovDecisionProcess(ANDMarkovDecisionProcess,
                                      TabularMarkovDecisionProcess):
    pass

