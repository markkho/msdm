from msdm.core.distributions import DictDistribution
import math


class SoftmaxDistribution(DictDistribution):
    def __init__(self, *args, **kwargs):
        """Creates normalized softmax distribution from element-score dictionary"""
        scores = dict(*args, **kwargs)
        max_score = max(scores.values())
        Z = sum([math.exp(s - max_score) for s in scores.values()])
        dist = {e: math.exp(s - max_score) / Z for e, s in scores.items()}
        super(SoftmaxDistribution, self).__init__(dist)
