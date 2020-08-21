import logging
from collections import defaultdict
import numpy as np
from scipy.special import softmax, logsumexp

np.seterr(divide='ignore')
logger = logging.getLogger(__name__)
logger.info("Ignoring division by zero errors")

from pyrlap.pyrlap2.core.distributions.distributions import Distribution

class Multinomial(Distribution):
    def __init__(self, support, logits=None, probs=None):
        if (probs is None) and (logits is None):
            logits = np.zeros(len(support))
        if len(support) == 0:
            probs = []
            logits = []
        if probs is None:
            if np.sum(logits) == -np.inf:
                probs = np.zeros(len(support))
            else:
                probs = softmax(logits)
        if logits is None:
            logits = np.log(probs)

        self._probs = tuple(probs)
        self._logits = tuple(logits)
        self._support = tuple(support)

    @property
    def support(self):
        return self._support

    def prob(self, e):
        try:
            return self._probs[self.support.index(e)]
        except ValueError:
            return 0

    @property
    def probs(self):
        #these are always normalized
        return self._probs

    def logit(self, e, default=-np.inf):
        try:
            return self._logits[self.support.index(e)]
        except ValueError:
            return default

    @property
    def logits(self):
        return self._logits

    def score(self, e):
        try:
            return self._logits[self.support.index(e)]
        except ValueError:
            return -np.inf

    def sample(self):
        if len(self.support) == 0:
            return
        return self.support[np.random.choice(len(self.support), p=self._probs)]

    def items(self, probs=False):
        if probs:
            return zip(self.support, self._probs)
        return zip(self.support, self._logits)

    def keys(self):
        return [e for e in self.support]

    def __len__(self):
        return len(self.support)

    def __str__(self):
        e_l = ", ".join([f"{e}: {l:.2f}" for e, l in self.items()])
        return f"{self.__class__.__name__}{{{e_l}}}"

    def __repr__(self):
        return str(self)

    def __and__(self, other: "Multinomial"):
        """Conjunction"""
        newdist = defaultdict(float)
        for e in self.support:
            newdist[e] += self.score(e)
        for e in other.support:
            newdist[e] += other.score(e)
        sup, scores = zip(*newdist.items())
        return Multinomial(support=sup, logits=scores)

    def __or__(self, other: "Multinomial"):
        """Disjunction/Mixture"""
        newdist = defaultdict(float)
        for e in self.support:
            newdist[e] += np.exp(self.score(e))
        for e in other.support:
            newdist[e] += np.exp(other.score(e))
        sup, scores = zip(*newdist.items())
        scores = [logsumexp(s) for s in scores]
        return Multinomial(support=sup, logits=scores)

    def __mul__(self, num):
        mlogits = [logit + np.log(num) for logit in self.logits]
        return Multinomial(support=self.support, logits=mlogits)

    def __rmul__(self, num):
        return self.__mul__(num)

    def __truediv__(self, num):
        mlogits = [logit - np.log(num) for logit in self.logits]
        return Multinomial(support=self.support, logits=mlogits)

    @property
    def Z(self):
        return np.exp(logsumexp(self.logits))

    def normalize(self):
        return self/self.Z

    def __sub__(self, other):
        """
        Distribution subtraction / negation
        """
        raise NotImplementedError

