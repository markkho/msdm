from abc import ABC, abstractmethod
from typing import Iterable
from msdm.core.problemclasses.problemclass import ProblemClass
from msdm.core.problemclasses.stochasticgame import TabularStochasticGame
from msdm.core.assignment.assignmentmap import AssignmentMap
from msdm.core.problemclasses.stochasticgame.policy.tabularpolicy import TabularMultiAgentPolicy,SingleAgentPolicy

class Algorithm(ABC):
    """Abstract superclass for all algorithms"""
    pass

class Result(ABC):
    """Abstract superclass for all result objects"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self):
        return '\n'.join(
            f'{repr(k)}: {repr(v)}'
            for k, v in vars(self).items()
        )

class Plans(Algorithm):
    @abstractmethod
    def plan_on(self, problem: ProblemClass) -> Result:
        pass

class Learns(Algorithm):
    @abstractmethod
    def train_on(self, problem: ProblemClass) -> Result:
        pass
    
