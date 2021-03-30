import random
import warnings
from collections import defaultdict
import math
from msdm.core.distributions.distributions import Distribution
import numpy as np


class DictDistribution(dict, Distribution):
    @classmethod
    def uniform(cls, support):
        p = 1/len(support)
        return DictDistribution({e: p for e in support})

    @classmethod
    def deterministic(cls, element):
        return DictDistribution({element: 1})

    def sample(self):
        if len(self.support) == 1:
            return self.support[0]
        return random.choices(
            population=self.support,
            weights=self.probs,
            k=1
        )[0]

    @property
    def support(self):
        try:
            return self._support
        except AttributeError:
            self._support = [e for e in self.keys() if self[e] > 0.0]
            return self._support

    @property
    def probs(self):
        try:
            return self._probs
        except AttributeError:
            self._probs = [self[k] for k in self.support]
            return self._probs

    def prob(self, e):
        return self.get(e, 0.0)

    def logit(self, e):
        p = self.prob(e)
        if p == 0:
            return -float('inf')
        return math.log(p)

    @property
    def logits(self):
        try:
            return self._logits
        except AttributeError:
            _logits = []
            for e in self.support:
                if self[e] == 0:
                    _logits.append(-float('inf'))
                else:
                    _logits.append(self.logit(e))
            self._logits = _logits
            return self._logits

    def score(self, e):
        return self.logit(e)

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

    def __repr__(self):
        e_p = ", ".join([f"{e}: {p}" for e, p in self.items()])
        return f"{self.__class__.__name__}({{{e_p}}})"

    def isclose(self, other):
        mapped = {
            s: p
            for s, p in zip(self.support, self.probs)
        }
        for s, p in zip(other.support, other.probs):
            if not np.isclose(p, mapped.get(s, 0.0)):
                return False
        return True

