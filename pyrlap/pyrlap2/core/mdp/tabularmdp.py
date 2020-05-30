from itertools import product
import numpy as np
from pyrlap.pyrlap2.core.mdp import ANDMarkovDecisionProcess, \
    MarkovDecisionProcess
from pyrlap.pyrlap2.core.variables import State, Action, TERMINALSTATE

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
            'rs': self.reachablestatevec
        }

    @property
    def states(self):
        try:
            return self._states
        except AttributeError:
            pass
        states = set([])
        statevars = tuple \
            ([v for v in self.variables if 'state' in v.properties])
        for values in product(*[v.domain for v in statevars]):
            states.add(State(statevars, tuple(values)))
        states.add(TERMINALSTATE)
        self._states = sorted(states)
        return self._states

    @property
    def actions(self):
        try:
            return self._actions
        except AttributeError:
            pass
        actions = set([])
        actionvars = tuple \
            ([v for v in self.variables if "action" in v.properties])
        for vals in product(*[v.domain for v in actionvars]):
            actions.add(Action(actionvars, tuple(vals)))
        self._actions = sorted(actions)
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
        for (si, ai, nsi), _ in np.ndenumerate(tf):
            s, a, ns = ss[si], aa[ai], ss[nsi]
            p = self.getNextStateDist(s, a).prob(ns)
            tf[si, ai, nsi] = p
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
        for (si, ai, nsi), _ in np.ndenumerate(rf):
            s, a, ns = ss[si], aa[ai], ss[nsi]
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
        self._ntvec = np.array([0 if s == TERMINALSTATE else 1 for s in ss])
        return self._ntvec

    @property
    def reachablestatevec(self):
        try:
            return self._reachablevec
        except AttributeError:
            pass
        s0 = self.getInitialStateDist().support
        frontier = set(s0)
        visited = set(s0)
        while len(frontier) > 0:
            s = frontier.pop()
            for a in self.getActionDist(s).support:
                for ns in self.getNextStateDist(s, a).support:
                    if ns not in visited:
                        frontier.add(ns)
                    visited.add(ns)
        self._reachablevec = np.array \
            ([1 if s in visited else 0 for s in self.states])
        return self._reachablevec

    def __and__(self, other: "TabularMarkovDecisionProcess"):
        assert isinstance(other, TabularMarkovDecisionProcess)
        assert all(v == u for v, u in zip(self.variables, other.variables)), \
            "Variables not aligned"
        assert all(s == z for s, z in zip(self.states, other.states)), \
            "State spaces not aligned"
        assert all(a == b for a, b in zip(self.actions, other.actions)), \
            "Action spaces not aligned"
        return ANDTabularMarkovDecisionProcess(self, other)

class ANDTabularMarkovDecisionProcess(ANDMarkovDecisionProcess,
                                      TabularMarkovDecisionProcess):
    pass

