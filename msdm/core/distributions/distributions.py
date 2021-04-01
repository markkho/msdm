from abc import ABC, abstractmethod
from typing import Iterable, Any
import random
import math

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

class FiniteDistribution(Distribution):
    @abstractmethod
    def prob(self, e) -> float:
        pass

    @property
    @abstractmethod
    def support(self) -> Iterable:
        pass

    def sample(self) -> Any:
        support = self.support
        if not isinstance(support, list):
            support = list(support)
        if len(support) == 1:
            return support[0]
        return random.choices(
            population=support,
            weights=list(self.probs),
            k=1
        )[0]

    def items(self):
        for e in self.support:
            yield e, self.prob(e)

    @property
    def probs(self):
        yield from (self.prob(e) for e in self.support)

    def score(self, e):
        p = self.prob(e)
        if p == 0:
            return -float('inf')
        return math.log(p)

    def __and__(self, other):
        pass

    def __or__(self, other):
        #can implement
        pass

    def __mul__(self, num):
        #can implement
        pass

    def __repr__(self):
        e_p = ", ".join([f"{e}: {p}" for e, p in self.items()])
        return f"{self.__class__.__name__}({{{e_p}}})"

    def isclose(self, other):
        mapped = {
            s: p
            for s, p in self.items()
        }
        for s, p in other.items():
            if not math.isclose(p, mapped.get(s, 0.0)):
                return False
        return True
