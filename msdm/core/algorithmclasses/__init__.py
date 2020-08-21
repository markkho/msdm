from abc import ABC, abstractmethod
from msdm.core.problemclasses.problemclass import ProblemClass

class Algorithm(ABC):
    """Abstract superclass for all algorithms"""
    pass

class Result(ABC):
    """Abstract superclass for all result objects"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

class Plans(Algorithm):
    @abstractmethod
    def planOn(self, problem: ProblemClass) -> Result:
        pass

class Learns(Algorithm):
    @abstractmethod
    def trainOn(self, problem: ProblemClass) -> Result:
        pass