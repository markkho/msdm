from abc import abstractmethod
from typing import Iterable

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution, DiscreteFactorTable
from msdm.core.assignment.assignmentcache import AssignmentCache


class MarkovDecisionProcess(ProblemClass):
    """
    MDPs are defined by:
    - action distributions, which bias actions at each state
    - initial state distributions
    - next state distributions
    """

    def __init__(self, memoize=True):
        if memoize:
            self.next_state_dist = AssignmentCache(self.next_state_dist)
            self.reward = AssignmentCache(self.reward)
            self.actions = AssignmentCache(self.actions)
            self.initial_state_dist = AssignmentCache(self.initial_state_dist)

    @abstractmethod
    def next_state_dist(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def reward(self, s, a, ns) -> float:
        pass

    @abstractmethod
    def actions(self, s) -> Iterable:
        pass

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def is_terminal(self, s):
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

    def next_state_dist(self, state, action) -> Distribution:
        d1 = self.mdp1.next_state_dist(state, action)
        d2 = self.mdp2.next_state_dist(state, action)
        return d1 & d2

    def reward(self, state, action, nextstate) -> float:
        r1 = self.mdp1.reward(state, action, nextstate)
        r2 = self.mdp2.reward(state, action, nextstate)
        return r1 + r2

    def actions(self, state) -> Iterable:
        #HACK: ideally this wouldn't need to convert to a distribution
        a1 = DiscreteFactorTable(self.mdp1.actions(state))
        a2 = DiscreteFactorTable(self.mdp2.actions(state))
        aa = a1 & a2
        return aa.support

    def initial_state_dist(self) -> Distribution:
        s1 = self.mdp1.initial_state_dist()
        s2 = self.mdp2.initial_state_dist()
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

    def next_state_dist(self, state, action) -> Distribution:
        d1 = self.mdp1.next_state_dist(state, action)
        d2 = self.mdp2.next_state_dist(state, action)
        return d1 | d2

    def reward(self, state, action, nextstate) -> float:
        r1 = self.mdp1.reward(state, action, nextstate)
        r2 = self.mdp2.reward(state, action, nextstate)
        assert r1 == r2, "Mixture of MDPs must have equivalent rewards" #may need to change
        return r1

    def actions(self, state) -> Distribution:
        #HACK: ideally this wouldn't need to convert to a distribution
        a1 = DiscreteFactorTable(self.mdp1.actions(state))
        a2 = DiscreteFactorTable(self.mdp2.actions(state))
        aa = a1 | a2
        return aa.support

    def initial_state_dist(self) -> Distribution:
        s1 = self.mdp1.initial_state_dist()
        s2 = self.mdp2.initial_state_dist()
        return s1 | s2


class ScaledMarkovDecisionProcess(MarkovDecisionProcess):
    """
    Base Scaled-MDP - only assumes function calls can be combined.

    This scales the logits of an MDP for disjunction with other MDPs.
    """
    def __init__(self, mdp, scale):
        self.mdp = mdp
        self.scale = scale
        MarkovDecisionProcess.__init__(self)

    def next_state_dist(self, s, a) -> Distribution:
        return self.mdp.next_state_dist(s, a) * self.scale

    def reward(self, state, action, nextstate) -> float:
        """rewards are unchanged"""
        return self.mdp.reward(state, action, nextstate)

    def actions(self, state) -> Iterable:
        return self.mdp.actions(state)

    def initial_state_dist(self) -> Distribution:
        return self.mdp.initial_state_dist() * self.scale
