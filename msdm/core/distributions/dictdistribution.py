from msdm.core.distributions.distributions import FiniteDistribution
import random


class UniformDistribution(FiniteDistribution):
    def __init__(self, support):
        self._support = support
    @property
    def support(self):
        return self._support
    def prob(self, e):
        if e in self._support:
            return 1/len(self.support)
        return 0
    def sample(self, *, rng=random):
        return rng.choice(self._support)

class DeterministicDistribution(FiniteDistribution):
    def __init__(self, value):
        self.value = value
    @property
    def support(self):
        return (self.value,)
    def prob(self, e):
        if e == self.value:
            return 1
        return 0
    def sample(self, *, rng=random):
        return self.value
    def items(self):
        yield self.value, 1

class DictDistribution(FiniteDistribution,dict):
    @classmethod
    def uniform(cls, support):
        return UniformDistribution(support)

    @classmethod
    def deterministic(cls, element):
        return DeterministicDistribution(element)

    @property
    def support(self):
        return self.keys()

    def prob(self, e):
        return self.get(e, 0.0)
