from abc import abstractmethod
from typing import Iterable

from pyrlap.core.problemclasses.problemclass import ProblemClass
from pyrlap.core.distributions import Distribution
from pyrlap.core.assignment.assignmentcache import AssignmentCache


class MarkovDecisionProcess(ProblemClass):
    """
    MDPs are defined by:
    - action distributions, which bias actions at each state
    - initial state distributions
    - next state distributions
    """

    def __init__(self, memoize=True):
        if memoize:
            self.getNextStateDist = AssignmentCache(self.getNextStateDist)
            self.getReward = AssignmentCache(self.getReward)
            self.getActions = AssignmentCache(self.getActions)
            self.getInitialStateDist = AssignmentCache(self.getInitialStateDist)

    @abstractmethod
    def getNextStateDist(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def getReward(self, s, a, ns) -> float:
        pass

    @abstractmethod
    def getActions(self, s) -> Iterable:
        pass

    @abstractmethod
    def getInitialStateDist(self) -> Distribution:
        pass

    @abstractmethod
    def isTerminal(self, s):
        pass

    def __and__(self, other: "MarkovDecisionProcess"):
        """
        This is not strictly an abstract base class method, but it
        should be overwritten by derived abstract base classes
        so that AND-composed MDPs can inherit from them.
        """
        return ANDMarkovDecisionProcess(self, other)

    def __or__(self, other: "MarkovDecisionProcess"):
        return ORMarkovDecisionProcess(self, other)

    def __mul__(self, num):
        return ScaledMarkovDecisionProcess(self, num)

    def __rmul__(self, num):
        return self.__mul__(num)


class ANDMarkovDecisionProcess(MarkovDecisionProcess):
    """Simplest AND MDP - only assumes function calls can be combined"""

    def __init__(self, mdp1, mdp2):
        self.mdp1 = mdp1
        self.mdp2 = mdp2
        MarkovDecisionProcess.__init__(self)

    def getNextStateDist(self, state, action) -> Distribution:
        d1 = self.mdp1.getNextStateDist(state, action)
        d2 = self.mdp2.getNextStateDist(state, action)
        return d1 & d2

    def getReward(self, state, action, nextstate) -> float:
        r1 = self.mdp1.getReward(state, action, nextstate)
        r2 = self.mdp2.getReward(state, action, nextstate)
        return r1 + r2

    def getActions(self, state) -> Iterable:
        a1 = self.mdp1.getActions(state)
        a2 = self.mdp2.getActions(state)
        #HACK: need standardized way of handling combination of actions
        for a in a1:
            assert a in a2, "Actions must match"
        for a in a2:
            assert a in a1, "Actions must match"
        return a1


    def getInitialStateDist(self) -> Distribution:
        s1 = self.mdp1.getInitialStateDist()
        s2 = self.mdp2.getInitialStateDist()
        return s1 & s2

class ORMarkovDecisionProcess(MarkovDecisionProcess):
    """
    Base OR-MDP - only assumes function calls can be combined.

    Note: The composed initial state, action, and transition 
    distributions are *mixtures* of the corresponding primitive
    distributions. That is, the index of the primitive MDP 
    is not represented. The reward functions of the corresponding
    primitive MDPs *must* equal one another.
    """

    #TODO: look into using the disjoint union of two MDPs as a more
    # general formulation
    def __init__(self, mdp1, mdp2):
        self.mdp1 = mdp1
        self.mdp2 = mdp2
        MarkovDecisionProcess.__init__(self)

    def getNextStateDist(self, state, action) -> Distribution:
        d1 = self.mdp1.getNextStateDist(state, action)
        d2 = self.mdp2.getNextStateDist(state, action)
        return d1 | d2

    def getReward(self, state, action, nextstate) -> float:
        r1 = self.mdp1.getReward(state, action, nextstate)
        r2 = self.mdp2.getReward(state, action, nextstate)
        assert r1 == r2, "Mixture of MDPs must have equivalent rewards" #may need to change
        return r1

    def getActions(self, state) -> Distribution:
        a1 = self.mdp1.getActions(state)
        a2 = self.mdp2.getActions(state)
        #HACK: need standardized way of handling combination of actions
        for a in a1:
            assert a in a2, "Actions must match"
        for a in a2:
            assert a in a1, "Actions must match"
        return a1

    def getInitialStateDist(self) -> Distribution:
        s1 = self.mdp1.getInitialStateDist()
        s2 = self.mdp2.getInitialStateDist()
        return s1 | s2


def ScaledMarkovDecisionProcess(MarkovDecisionProcess):
    """
    Base Scaled-MDP - only assumes function calls can be combined.

    This scales the logits of an MDP for disjunction with other MDPs.
    """
    def __init__(self, mdp, scale):
        self.mdp = mdp
        self.scale = scale
        MarkovDecisionProcess.__init__(self)

    def getNextStateDist(self, state, action) -> Distribution:
        return self.mdp.getNextStateDist(state, action)*self.scale

    def getReward(self, state, action, nextstate) -> float:
        """rewards are unchanged"""
        return self.mdp.getReward(state, action, nextstate)

    def getActionDist(self, state) -> Distribution:
        return self.mdp.getActionDist(state)*self.scale

    def getInitialStateDist(self) -> Distribution:
        return self.mdp.getInitialStateDist()*self.scale
