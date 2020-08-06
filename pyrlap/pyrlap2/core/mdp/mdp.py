from typing import Iterable
from abc import ABC, abstractmethod

from pyrlap.pyrlap2.core.variables import TaskVariable, State, Action
from pyrlap.pyrlap2.core.distributions import Distribution

class MarkovDecisionProcess(ABC):
    """
    MDPs are defined by:
    - action distributions, which bias actions at each state
    - initial state distributions
    - next state distributions
    """

    def __init__(self):
        pass

    @abstractmethod
    def getNextStateDist(self, s: State, a: Action) -> Distribution:
        pass

    @abstractmethod
    def getReward(self, s: State, a: Action, ns: State) -> float:
        pass

    @abstractmethod
    def getActionDist(self, s: State) -> Distribution:
        pass

    @abstractmethod
    def getInitialStateDist(self) -> Distribution:
        pass

    def __and__(self, other: "MarkovDecisionProcess"):
        """
        This is not strictly an abstract base class method, but it
        should be overwritten by derived abstract base classes
        so that AND-composed MDPs can inherit from them.
        """
        return ANDMarkovDecisionProcess(self, other)


class ANDMarkovDecisionProcess(MarkovDecisionProcess):
    """Simplest AND MDP - only assumes function calls can be combined"""

    def __init__(self, mdp1, mdp2):
        self.mdp1 = mdp1
        self.mdp2 = mdp2
        variables = sorted(set(mdp1.variables + mdp2.variables))
        MarkovDecisionProcess.__init__(self, variables)

    def getNextStateDist(self, state, action) -> Distribution:
        d1 = self.mdp1.getNextStateDist(state, action)
        d2 = self.mdp2.getNextStateDist(state, action)
        return d1 & d2

    def getReward(self, state, action, nextstate) -> float:
        r1 = self.mdp1.getReward(state, action, nextstate)
        r2 = self.mdp2.getReward(state, action, nextstate)
        return r1 + r2

    def getActionDist(self, state) -> Distribution:
        a1 = self.mdp1.getActionDist(state)
        a2 = self.mdp2.getActionDist(state)
        return a1 & a2

    def getInitialStateDist(self) -> Distribution:
        s1 = self.mdp1.getInitialStateDist()
        s2 = self.mdp2.getInitialStateDist()
        return s1 & s2