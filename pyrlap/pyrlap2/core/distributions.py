from abc import ABC, abstractmethod
from itertools import product
from copy import deepcopy
from collections import defaultdict
import numpy as np
from scipy.special import softmax
from pyrlap.pyrlap2.core.enumerable import Enumerable
from pyrlap.pyrlap2.core.utils import dict_merge, dict_match, naturaljoin

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

    def items(self, probs=False):
        if probs:
            return zip(self.support, self._probs)
        return zip(self.support, self._logits)

    def keys(self):
        return [e for e in self.support]

    def asMatrix(self, rep="logits"):
        if rep == 'logits':
            return np.array(self._logits)
        return np.array(self._probs)

    def __and__(self, other: "Multinomial"):
        """
        Distribution conjunction:
        Multiplies two multinomial distributions (adds their log probabilities) 
        with optionally factored variables.
        If the support of both are dictionaries, each nested key functions
        as a variable name. If there are shared variables, these are combined
        by adding the energies of support elements where their assignments match, 
        effectively making each distribution a factor in a factor graph.
        """
        jsupport = []
        jlogits = []
        for si, oi in product(self.support, other.support):
            if isinstance(si, dict) and isinstance(oi, dict):
                if dict_match(si, oi): #not efficient if the cartesian product is large 
                    jsupport.append(dict_merge(si, oi))
                    jlogits.append(self.logit(si) + other.logit(oi))
            else:
                jsupport.append((si, oi))
                jlogits.append(self.logit(si) + other.logit(oi))
        assert len(jlogits) > 0, "Degenerate distribution"
        return Multinomial(support=jsupport, logits=jlogits)

    def __or__(self, other):
        """
        Distribution disjunction
        This is equivalent to marginalization
        see https://arxiv.org/pdf/2004.06030.pdf
        """
        raise NotImplementedError

    def __sub__(self, other):
        """
        Distribution subtraction / negation
        """
        raise NotImplementedError

    def __str__(self):
        e_l = ", ".join([f"{e}: {l:.2f}" for e, l in self.items()])
        return f"{self.__class__.__name__}{{{e_l}}}"

    def __repr__(self):
        return str(self)