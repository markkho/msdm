from abc import ABC, abstractmethod
from typing import Sequence, Any, TypeVar, Generic, Tuple, Callable, Union, Dict
import random
import math
from collections import defaultdict

from msdm.core.utils.funcutils import cached_property

Event = TypeVar('Event')

class Distribution(ABC, Generic[Event]):
    @abstractmethod
    def sample(self, *, rng=random) -> Event:
        pass

    def marginalize(self, projection: Callable[[Event], Event]) -> "Distribution":
        raise NotImplementedError
    
    def condition(self, predicate: Callable[[Event], Union[bool, float]]) -> "Distribution":
        raise NotImplementedError
    
    def expectation(self, real_function: Callable[[Event], float] = lambda e: e) -> float:
        raise NotImplementedError
    
    def chain(self, function: Callable[[Event], "Distribution"]) -> "Distribution":
        raise NotImplementedError
    
    def score(self, e: Event) -> float:
        raise NotImplementedError
    
    def support(self) -> Sequence[Event]:
        raise NotImplementedError

class ImplicitDistribution(Distribution):
    def __init__(
        self,
        stochastic_function : Callable[[random.Random], Event],
        n_samples : int,
        _seed : int = None
    ):
        self.stochastic_function = stochastic_function
        self.n_samples = n_samples
        self._seed = _seed
    
    @cached_property
    def _rng(self):
        return random.Random(self._seed)

    def sample(self, *, rng=None) -> Event:
        if rng is None:
            rng = self._rng
        return self.stochastic_function(rng)
    
    @property
    def _monte_carlo_simulation(self) -> Dict[Event, float]:
        counts = defaultdict(int)
        rng = self._rng
        for _ in range(self.n_samples):
            counts[self.sample(rng=rng)] += 1
        counts = {e: c / self.n_samples for e, c in counts.items()}
        return counts
    
    def items(self):
        yield from self._monte_carlo_simulation.items()

    def marginalize(self, projection : Callable[[Any], Any]) -> "ImplicitDistribution":
        def projected_function(rng : random.Random):
            return projection(self.stochastic_function(rng))
        return ImplicitDistribution(
            projected_function,
            n_samples=self.n_samples,
            _seed=self._seed
        )

    def condition(self, predicate : Callable[[Any], bool]) -> "ImplicitDistribution":
        def rejection_sampler(rng : random.Random):
            for _ in range(self.n_samples):
                sample = self.stochastic_function(rng)
                if predicate(sample):
                    return sample
            raise ValueError(f"No sample (n = {self.n_samples}) satisfies the predicate")
        return ImplicitDistribution(
            rejection_sampler,
            n_samples=self.n_samples,
            _seed=self._seed
        )

    def expectation(self, real_function : Callable[[Any], float] = lambda e: e) -> float:
        rng = self._rng
        val = 0
        for _ in range(self.n_samples):
            val += real_function(self.stochastic_function(rng))
        return val / self.n_samples

class FiniteDistribution(Distribution[Event]):
    @abstractmethod
    def prob(self, e: Event) -> float:
        pass

    @property
    @abstractmethod
    def support(self) -> Sequence[Event]:
        pass

    def __len__(self):
        return len(self.support)

    def sample(self, *, rng=random, k=1) -> Event:
        support = self.support
        if not isinstance(support, (list, tuple)):
            support = tuple(support)
        if len(support) == 1:
            return support[0]
        s = rng.choices(
            population=support,
            weights=tuple(self.probs),
            k=k
        )
        if k == 1:
            return s[0]
        return s

    def items(self) -> Sequence[Tuple[Event, float]]:
        for e in self.support:
            yield e, self.prob(e)

    def values(self) -> Sequence[float]:
        for e in self.support:
            yield self.prob(e)

    @property
    def probs(self) -> Sequence[float]:
        yield from (self.prob(e) for e in self.support)

    def score(self, e: Event) -> float:
        p = self.prob(e)
        if p == 0:
            return -float('inf')
        return math.log(p)

    def __and__(self, other: "FiniteDistribution[Event]") -> "FiniteDistribution[Event]":
        """Conjunction"""
        newdist = defaultdict(float)
        norm = 0.
        for e in set(self.support) & set(other.support):
            newdist[e] += self.score(e)
            newdist[e] += other.score(e)
            norm += math.exp(newdist[e])
        lognorm = math.log(norm)
        return DictDistribution({e: math.exp(l - lognorm) for e, l in newdist.items()})

    def __or__(self, other: "FiniteDistribution[Event]") -> "FiniteDistribution[Event]":
        """Disjunction/Mixture"""
        newdist = defaultdict(float)
        for e, p in self.items():
            newdist[e] += p
        for e, p in other.items():
            newdist[e] += p
        return DictDistribution(newdist)

    def __mul__(self, num: float):
        return DictDistribution({e: p*num for e, p in self.items()})

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        e_p = ", ".join([f"{e}: {p}" for e, p in self.items()])
        return f"{self.__class__.__name__}({{{e_p}}})"

    def isclose(
        self, other: "FiniteDistribution[Event]", *,
        # These tolerances are copied from `np.isclose()`
        rtol=1e-05, atol=1e-08,
    ) -> bool:
        isclose_kw = dict(rel_tol=rtol, abs_tol=atol)
        nonzero_support = {
            e
            for dist in [self, other]
            for e, p in dist.items()
            if not math.isclose(p, 0, **isclose_kw)
        }
        for e in nonzero_support:
            if not math.isclose(self.prob(e), other.prob(e), **isclose_kw):
                return False
        return True

    def marginalize(self, projection: Callable[[Event], Event]):
        """
        Marginalize the distribution according to the
        projection function. `projection` must return hashable values.
        """
        newdist = defaultdict(lambda : 0)
        for e, p in self.items():
            newdist[projection(e)] += p
        return DictDistribution(newdist)

    def expectation(self, real_function: Callable[[Event], float] = lambda e: e):
        """
        Return the expected value of real_function under
        the distribution.
        """
        tot = 0
        for e, p in self.items():
            tot += real_function(e)*p
        return tot

    def condition(self, predicate: Callable[[Event], Union[bool, float]]):
        """
        Given a function that returns probabilities for
        each element, return a new *normalized* distribution
        with initial probabilities multiplied by function probabilities.

        This is useful for calculating a posterior distribution. E.g.
        ```
        prior_x = DictDistribution(a=.5, b=.5)
        likelihood_x = lambda x : {'a': .2, 'b': .5}[x]
        posterior_x = prior.condition(likelihood_x)
        ```
        """
        dist = {}
        norm = 0
        for e, p in self.items():
            weight = predicate(e)
            if weight > 0:
                dist[e] = p*weight
                norm += dist[e]
        dist = {e: p/norm for e, p in dist.items()}
        return DictDistribution(dist)

    def chain(self, function: Callable[[Event], Distribution]) -> Distribution:
        """
        Chain a function f that returns a new distribution over Y, given
        an element from the current support X [e.g., y ~ f(x)].
        The final distribution corresponds to the
        joint distribution with the "prior"
        variables marginalized out [i.e., p(y) = sum_x(p_f(y | x)p(x))].
        """
        cum_dist = defaultdict(float)
        for e, p in self.items():
            new_dist = function(e)
            for new_e, new_p in new_dist.items():
                cum_dist[new_e] += p*new_p
        return DictDistribution(cum_dist)

    def normalize(self):
        total = sum(self.values())
        return DictDistribution({e: p/total for e, p in self.items()})

    def joint(self, other: "FiniteDistribution"):
        return DictDistribution({
            (a, b): pa * pb
            for a, pa in self.items()
            for b, pb in other.items()
        })

    def is_normalized(self, rtol=1e-05, atol=1e-08):
        return math.isclose(sum(self.probs), 1, rel_tol=rtol, abs_tol=atol)

# Importing down here to avoid a cyclic reference.
from msdm.core.distributions.dictdistribution import DictDistribution
