from abc import ABC, abstractmethod
from pyrlap.pyrlap2.core.problemclasses.problemclass import ProblemClass

class Algorithm(ABC):
    """Abstract superclass for all algorithms"""
    pass

class Plans(Algorithm):
    @abstractmethod
    def planOn(self, problem: ProblemClass):
        pass

class Learns(Algorithm):
    @abstractmethod
    def trainOn(self, problem: ProblemClass):
        pass