from msdm.core.distributions import FiniteDistribution
import math


class SoftmaxDistribution(FiniteDistribution):
    def __init__(self, *args, **kwargs):
        """Creates normalized softmax distribution from element-score dictionary"""
        scores = dict(*args, **kwargs)
        max_score = max(scores.values())
        Z = sum([math.exp(s - max_score) for s in scores.values()])
        self._dist = {e: math.exp(s - max_score) / Z for e, s in
                      scores.items()}

    @property
    def support(self):
        return self._dist.keys()

    def prob(self, e):
        return self._dist.get(e, 0.0)

    def __repr__(self):
        e_p = ", ".join(
            [f"{e.__repr__()}: {math.log(p).__repr__()}" for e, p in
             self.items()])
        return f"{self.__class__.__name__}({{{e_p}}})"


