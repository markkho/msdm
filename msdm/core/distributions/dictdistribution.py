from collections import defaultdict
from msdm.core.distributions.distributions import FiniteDistribution
import math


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

class DeterministicDistribution(FiniteDistribution):
    def __init__(self, value):
        self.value = value
    @property
    def support(self):
        return [self.value]
    def prob(self, e):
        if e == self.value:
            return 1
        return 0

class DictDistribution(dict, FiniteDistribution):
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

    def __and__(self, other: "DictDistribution"):
        """Conjunction"""
        newdist = defaultdict(float)
        norm = 0
        for e in set(self.support) & set(other.support):
            newdist[e] += self.score(e)
            newdist[e] += other.score(e)
            norm += math.exp(newdist[e])
        lognorm = math.log(norm)
        return DictDistribution({e: math.exp(l - lognorm) for e, l in newdist.items()})

    def __or__(self, other: "DictDistribution"):
        """Disjunction/Mixture"""
        newdist = defaultdict(float)
        for e in self.support:
            newdist[e] += self[e]
        for e in other.support:
            newdist[e] += other[e]
        return DictDistribution(newdist)

    def __mul__(self, num):
        return DictDistribution({e: p*num for e, p in dict.items(self)})
