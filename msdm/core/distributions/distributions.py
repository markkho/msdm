from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    @property
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def prob(self, e):
        pass

    @abstractmethod
    def score(self, e):
        pass

class DiscreteDistribution(Distribution):
    @abstractmethod
    def items(self): #yields elements and probabilities
        pass

    @property
    @abstractmethod
    def probs(self):
        pass

