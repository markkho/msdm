from typing import Mapping, Hashable
from abc import abstractmethod

from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.distributions import Distribution
from msdm.core.assignment.assignmentcache import AssignmentCache

class StochasticGame(ProblemClass):
    """
    SGs are defined by:
    - action distributions, which bias actions at each state
    - initial state distributions
    - next state distributions
    """

    def __init__(self, agent_names, memoize=True):
        self._agentNames = agent_names
        if memoize:
            self.next_state_dist = AssignmentCache(self.next_state_dist)
            self.joint_rewards = AssignmentCache(self.joint_rewards)
            self.joint_action_dist = AssignmentCache(self.joint_action_dist)
            self.initial_state_dist = AssignmentCache(self.initial_state_dist)
            self.is_terminal = AssignmentCache(self.is_terminal)
        
    @property
    def agent_names(self):
        return self._agentNames

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def joint_action_dist(self, s) -> Distribution:
        # Deprecated 
        pass
    
    @abstractmethod 
    def joint_actions(self,s):
        # Dict from agent -> actions 
        pass 

    @abstractmethod
    def is_terminal(self, s) -> bool:
        pass

    @abstractmethod
    def next_state_dist(self, s, ja) -> Distribution:
        """Joint action should be dictionary with agent names as keys"""
        pass

    @abstractmethod
    def joint_rewards(self,
                      s,  # state
                      ja,  # jointaction
                      ns,  # nextstate
                      ) -> Mapping[Hashable, float]:
        """This should return a mapping from agent names to rewards"""
        pass

    def __and__(self, other: "StochasticGame"):
        """
        Specific implementations may want to overwrite this.
        """
        return ANDStochasticGame(self, other)


class ANDStochasticGame(StochasticGame):
    """Simplest AND SG - only assumes function calls can be combined"""

    def __init__(self, sg1, sg2):
        raise NotImplementedError
    #     self.mdp1 = mdp1
    #     self.mdp2 = mdp2
    #     variables = sorted(set(mdp1.variables + mdp2.variables))
    #     MarkovDecisionProcess.__init__(self, variables)

    # def getNextStateDist(self, state, action) -> Distribution:
    #     d1 = self.mdp1.getNextStateDist(state, action)
    #     d2 = self.mdp2.getNextStateDist(state, action)
    #     return d1 & d2

    # def getReward(self, state, action, nextstate) -> float:
    #     r1 = self.mdp1.getReward(state, action, nextstate)
    #     r2 = self.mdp2.getReward(state, action, nextstate)
    #     return r1 + r2

    # def getActionDist(self, state) -> Distribution:
    #     a1 = self.mdp1.getActionDist(state)
    #     a2 = self.mdp2.getActionDist(state)
    #     return a1 & a2

    # def getInitialStateDist(self) -> Distribution:
    #     s1 = self.mdp1.getInitialStateDist()
    #     s2 = self.mdp2.getInitialStateDist()
    #     return s1 & s2
