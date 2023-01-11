from typing import Mapping, Hashable
from abc import abstractmethod

from msdm.core.distributions import Distribution

class StochasticGame:
    def __init__(self, agent_names):
        self.agent_names = agent_names

    @abstractmethod
    def initial_state_dist(self) -> Distribution:
        pass

    @abstractmethod 
    def joint_actions(self,s) -> dict:
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
    def joint_rewards(self, s, ja,  ns) -> Mapping[Hashable, float]:
        """This should return a mapping from agent names to rewards"""
        pass

