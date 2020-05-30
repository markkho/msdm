from abc import ABC, abstractmethod
from typing import Mapping
import numpy as np

from pyrlap.pyrlap2.core.agent.agent import Agent
from pyrlap.pyrlap2.core.mdp.mdp import MarkovDecisionProcess
class TabularAgent(Agent):
    def evaluateOn(self, mdp: MarkovDecisionProcess) -> Mapping:
        # do policy evaluation
        raise NotImplementedError

    @property
    @abstractmethod
    def valuefunc(self):
        pass

    @property
    @abstractmethod
    def actionvaluefunc(self):
        pass

    def asDict(self):
        policydict = {}
        for s in self.states:
            policydict[s] = {}
            for a in self.actions:
                policydict[s][a] = self.getActionDist(s).prob(a)
        return policydict

    @property
    def states(self):
        try:
            return self._states
        except AttributeError:
            raise AttributeError(
                "Agent does not have a state space `._states`")

    @property
    def actions(self):
        try:
            return self._actions
        except AttributeError:
            raise AttributeError(
                "Agent does not have an action space `._actions`")

    @property
    def policymat(self):
        try:
            return self._policymat
        except AttributeError:
            pass
        ss = self.states
        aa = self.actions
        pi = np.zeros(len(ss), len(aa))
        for (si, ai), _ in np.ndenumerate(pi):
            s, a = ss[si], aa[ai]
            p = self.getActionDist(s).prob(a)
            pi[si, ai] = p
        self._policymat = pi
        return self._policymat

    @property
    def qvaluemat(self):
        return self._qvaluemat

    @property
    def valuevec(self):
        return self._valuevec
