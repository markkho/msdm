from typing import Mapping, Hashable
from abc import abstractmethod

from msdm.core.distributions import Distribution


class PartiallyObservableStochasticGame:
    """
    POSGs are defined by:
    - action distributions, which bias actions at each state
    - initial state distributions
    - next state, observation distributions
    """

    def __init__(self, agent_names):
        self._agentNames = agent_names

    @property
    def agent_names(self):
        return self._agentNames

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod
    def joint_action_dist(self, s) -> Distribution:
        pass

    @abstractmethod
    def next_state_dist(self,
                        s,  # state
                        ja,  # jointaction
                        ) -> Distribution:
        pass

    @abstractmethod
    def joint_observation_dist(self,
                               s,  # state
                               ja,  # jointaction
                               ns,  # nextstate
                               ) -> Distribution:
        pass

    @abstractmethod
    def joint_rewards(self,
                      s,  # state
                      ja,  # jointaction
                      ns,  # nextstate
                      jo  # jointobservation
                      ) -> Mapping[Hashable, float]:
        pass

    def __and__(self, other: "PartiallyObservableStochasticGame"):
        """
        This should be overwritten for specific implementations
        """
        return ANDPartiallyObservableStochasticGame(self, other)


class ANDPartiallyObservableStochasticGame(PartiallyObservableStochasticGame):
    """Simplest AND POSG - only assumes function calls can be combined"""

    def __init__(self, posg1, posg2):
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
