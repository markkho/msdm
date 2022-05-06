import functools
from msdm.core.utils.funcutils import method_cache
from msdm.core.distributions.distributions import FiniteDistribution
from msdm.core.distributions.ppl.reify import reify

class LazyFunctionalDistribution(FiniteDistribution):
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
    @method_cache
    def reified_dist(self):
        return reify(self.function)(*self.args, **self.kwargs)
    def sample(self, *, rng=None):
        if rng is not None:
            return self.function(*self.args, **self.kwargs, rng=rng)
        return self.function(*self.args, **self.kwargs)
    def prob(self, e):
        return self.reified_dist().prob(e)
    def __len__(self):
        return self.reified_dist().__len__()
    @property
    def support(self):
        return self.reified_dist().support
    __repr__ = object.__repr__

def lazy_distribution_generator(f):
    @functools.wraps(f)
    @functools.lru_cache()
    def dist_gen(*args, **kwargs):
        return LazyFunctionalDistribution(f, *args, **kwargs)
    return dist_gen
