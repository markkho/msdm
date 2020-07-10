from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from scipy.special import softmax
from pyrlap.pyrlap2.core.enumerable import Enumerable

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def prob(self, e):
        pass

    @abstractmethod
    def logit(self, e):
        pass

    @abstractmethod
    def __and__(self, other):
        pass



class Multinomial(Enumerable, Distribution):
    def __init__(self, support, logits=None, probs=None):
        if (probs is None) and (logits is None):
            logits = np.zeros(len(support))
        probs = probs if probs is not None else softmax(logits)
        logits = logits if logits is not None else np.log(probs)

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
        return self._probs

    def logit(self, e):
        try:
            return self._logits[self.support.index(e)]
        except ValueError:
            return -np.inf

    @property
    def logits(self):
        return self._logits

    def sample(self):
        return self.support[np.random.choice(len(self.support), p=self._probs)]

    def items(self):
        return zip(self.support, self._logits)

    def keys(self):
        return [e for e in self.support]

    def asMatrix(self, rep="logits"):
        if rep == 'logits':
            return np.array(self._logits)
        return np.array(self._probs)

    def combineEnergyWith(self, other: "Multinomial",
                          selfweight=1.0,
                          otherweight=1.0):
        """
        Combines two distributions by multiplying probabilities
        and normalizing
        """
        fullsupport = sorted(set(self.support + other.support))
        new_logits = {e: self.logit(e)*selfweight +
                         other.logit(e)*otherweight
                      for e in fullsupport}
        new_logits = {e: l for e, l in new_logits.items() if l != -np.inf}
        assert len(new_logits) > 0, "Degenerate distribution"
        new_support, new_logits = zip(*sorted(new_logits.items()))
        return Multinomial(support=new_support, logits=new_logits)

    def mixWith(self, other: "Multinomial", selfprob: float = .5):
        """
        Returns the distribution associated with a weighted mixture
        of self and other.
        """
        if selfprob == 0.0:
            return other
        elif selfprob == 1.0:
            return self

        new_probs = defaultdict(float)
        for e, p in zip(self.support, self.probs):
            new_probs[e] += p*selfprob
        for e, p in zip(other.support, other.probs):
            new_probs[e] += p*(1-selfprob)
        new_support, new_probs = zip(*sorted(new_probs.items()))
        return Multinomial(support=new_support, probs=new_probs)

    def __and__(self, other: "Multinomial"):
        return self.combineEnergyWith(other)

    def __str__(self):
        e_l = ", ".join([f"{e}: {l:.2f}" for e, l in self.items()])
        return f"{self.__class__.__name__}{{{e_l}}}"

    def __repr__(self):
        return str(self)