from msdm.core.distributions.distributions import FiniteDistribution
from collections import defaultdict
import random


class UniformDistribution(FiniteDistribution):
    def __init__(self, support, check_unique=True):
        if check_unique:
            assert len(support) == len(set(support)), (
                f'Invalid support {support}: some event is duplicated.'
            )
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

class DictDistribution(dict,FiniteDistribution):
    @classmethod
    def uniform(cls, support):
        return UniformDistribution(support)

    @classmethod
    def deterministic(cls, element):
        return DeterministicDistribution(element)

    @classmethod
    def from_pairs(cls, element_probs):
        dist = defaultdict(float)
        for e, p in element_probs:
            dist[e] += p
        return DictDistribution(dist)

    @property
    def support(self):
        return self.keys()

    def prob(self, e):
        return self.get(e, 0.0)

    items = dict.items
    values = dict.values
    __or__ = FiniteDistribution.__or__
    __and__ = FiniteDistribution.__and__
    __mul__ = FiniteDistribution.__mul__
    __rmul__ = FiniteDistribution.__rmul__
    __repr__ = FiniteDistribution.__repr__
