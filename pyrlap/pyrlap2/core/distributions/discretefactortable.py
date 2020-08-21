import logging
from itertools import product
import numpy as np
from scipy.special import softmax, logsumexp
from typing import Mapping

np.seterr(divide='ignore')
logger = logging.getLogger(__name__)
logger.info("Ignoring division by zero errors")

from pyrlap.pyrlap2.core.utils.dictutils import dict_merge, dict_match
from pyrlap.pyrlap2.core.distributions.distributions import Distribution

class DiscreteFactorTable(Distribution):
    """
    Discrete factor table assigns a score to variable assignments. If support
    elements are (nested) mappings, each (nested) key is interpreted as
    a variable and corresponding value is an assignment. If the elements are
    non-mappings, then the distribution behaves like a Multinomial.
    """
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

    def __and__(self, other: "DiscreteFactorTable"):
        """
        Distribution conjunction:
        Multiplies two multinomial distributions (adds their log probabilities)
        with optionally factored variables.
        If the support of both are dictionaries, each nested key functions
        as a variable name. If there are shared variables, these are combined
        by adding the energies of support elements where their assignments match,
        effectively making each distribution a factor in a factor graph.
        """

        #Conjunction with null distribution is null distribution
        if (len(self.support) == 0) or (len(other.support) == 0):
            return DiscreteFactorTable([])

        jsupport = []
        jlogits = []
        # HACK: this should throw a warning or something if the distributions have different headers
        # i.e., dictionary keys interact in a weird way
        for si, oi in product(self.support, other.support):
            if isinstance(si, dict) and isinstance(oi, dict):
                if dict_match(si, oi): #not efficient if the cartesian product is large
                    soi = dict_merge(si, oi)
                    if soi in jsupport:
                        continue
                    logit = self.logit(si) + other.logit(oi)
                    if logit == -np.inf:
                        continue
                    jsupport.append(soi)
                    jlogits.append(logit)
            else:
                jsupport.append((si, oi))
                jlogits.append(self.logit(si) + other.logit(oi))
        if len(jlogits) > 0:
            logger.debug("Product distribution has no non-zero support")
        return DiscreteFactorTable(support=jsupport, logits=jlogits)

    def __or__(self, other):
        """
        Mixture of experts

        A mixture of experts energy can be calculated from primitive
        energy functions p and q by taking log(exp(p(x)) + exp(q(x))).

        This handles dictionaries as rows of factor tables. Care should be taken
        to ensure that each dictionary contains the relevant keys since
        otherwise the energies will be combined in unexpected ways.

        Note that the logits will not be normalized.

        Example:

        pqequalmix = p | q
        pqunequalmix = p*.2 | q*.8
        pqunequalmix = p*.1 | q*.8 | p*.1
        """
        if (len(self.support) == 0):
            return other
        if (len(other.support) == 0):
            return self

        jsupport = []
        jlogits = []
        matchedrows = []
        unmatchedrows = []
        # HACK: this should throw a warning or something if the distributions have different headers
        # i.e., dictionary keys interact in a weird way

        #first get inner join rows, tracking ones that don't match
        for si, oi in product(self.support, other.support):
            if isinstance(si, dict) and isinstance(oi, dict):
                if dict_match(si, oi): #not efficient if the cartesian product is large
                    matchedrows.extend([si, oi])
                    soi = dict_merge(si, oi)
                    if soi in jsupport:
                        continue
                    logit = self.logit(si) + other.logit(oi)
                    if logit == -np.inf:
                        continue
                    jsupport.append(soi)
                    jlogits.append(logit)
                else:
                    unmatchedrows.extend([si, oi])
            else:
                jsupport.append((si, oi))
                jlogits.append(self.logit(si) + other.logit(oi))

        #add in the left and right outer join rows, ensuring that they were never matched
        for i in unmatchedrows:
            if (i in matchedrows) or (i in jsupport):
                continue
            logit = np.log(np.exp(self.logit(i)) + np.exp(other.logit(i)))
            if (logit == -np.inf):
                continue
            jsupport.append(i)
            jlogits.append(logit)
        return DiscreteFactorTable(support=jsupport, logits=jlogits)

    def __mul__(self, num):
        mlogits = [logit + np.log(num) for logit in self.logits]
        return DiscreteFactorTable(support=self.support, logits=mlogits)

    def __rmul__(self, num):
        return self.__mul__(num)

    def __truediv__(self, num):
        mlogits = [logit - np.log(num) for logit in self.logits]
        return DiscreteFactorTable(support=self.support, logits=mlogits)

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

    def __str__(self):
        e_l = ", ".join([f"{e}: {l:.2f}" for e, l in self.items()])
        return f"{self.__class__.__name__}{{{e_l}}}"

    def __repr__(self):
        return str(self)
