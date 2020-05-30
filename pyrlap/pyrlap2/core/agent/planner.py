from abc import abstractmethod
from pyrlap.pyrlap2.core.agent.agent import Agent

class Planner(Agent):
    @abstractmethod
    def planOn(self, mdp):
        pass
