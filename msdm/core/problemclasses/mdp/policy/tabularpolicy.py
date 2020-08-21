from typing import Mapping
import numpy as np

from msdm.core.problemclasses.mdp.policy.policy import Policy
from msdm.core.problemclasses.mdp.mdp import MarkovDecisionProcess

from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.distributions import DiscreteFactorTable, Distribution
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
        return DiscreteFactorTable(support=a, probs=p)

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

