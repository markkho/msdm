from typing import Mapping
import numpy as np

from pyrlap.pyrlap2.core.policy.policy import Policy
from pyrlap.pyrlap2.core.assignmentmap import AssignmentMap
from pyrlap.pyrlap2.core.distributions import Multinomial, Distribution
from pyrlap.pyrlap2.core.mdp.mdp import MarkovDecisionProcess
class TabularPolicy(Policy):
    def __init__(self, states, actions, policymatrix=None, policydict=None):
        self._states = states
        self._actions = actions
        if policymatrix is not None:
            policydict = AssignmentMap()
            for si, s in enumerate(states):
                policydict[s] = AssignmentMap()
                for ai, a in enumerate(actions):
                    if policymatrix[si, ai] > 0:
                        policydict[s][a] = policymatrix[si, ai]
        self._policydict = policydict

    def evaluateOn(self, mdp: MarkovDecisionProcess) -> Mapping:
        # do policy evaluation
        raise NotImplementedError

    def getActionDist(self, s) -> Distribution:
        adist = self._policydict[s]
        a, p = zip(*adist.items())
        return Multinomial(support=a, probs=p)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def policydict(self) -> Mapping:
        return self._policydict

    @property
    def policymat(self):
        try:
            return self._policymat
        except AttributeError:
            pass
        pi = np.zeros((len(self.states), len(self.actions)))
        for si, s in enumerate(self.states):
            adist = self.getActionDist(s)
            for ai, a in enumerate(self.actions):
                pi[si, ai] = adist.prob(a)
        self._policymat = pi
        return self._policymat

    # @property
    # def qvaluemat(self):
    #     return self._qvaluemat
    #
    # @property
    # def valuevec(self):
    #     return self._valuevec
