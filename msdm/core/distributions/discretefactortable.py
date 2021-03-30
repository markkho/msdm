import logging
import warnings
from itertools import product
import numpy as np
from scipy.special import softmax, logsumexp

np.seterr(divide='ignore')
logger = logging.getLogger(__name__)
logger.info("Ignoring division by zero errors")

from msdm.core.utils.dictutils import dict_merge, dict_match
from msdm.core.distributions.distributions import Distribution
from msdm.core.assignment import DefaultAssignmentMap

from frozendict import frozendict

class DiscreteFactorTable(Distribution):
    """
    A discrete factor table maps variable assignments to real-valued scores.
    If support elements are (nested) mappings, each (nested) key is interpreted as
    a variable and corresponding value is an assignment. If the elements are
    non-mappings, then the distribution behaves like a Multinomial.
    """
    def __init__(self, support, logits=None, probs=None, scores=None):
        if scores is None:
            scores = logits
        if isinstance(support, dict):
            assert logits is None
            assert probs is None
            support, scores = zip(*support.items())
        if len(support) == 0:
            probs = []
            scores = []
        if (probs is None) and (scores is None):
            scores = (0,)*len(support)
            probs = (1/len(support),)*len(support)
        if probs is None:
            assert len(support) == len(scores)
            if np.sum(scores) == -np.inf:
                probs = np.zeros(len(support))
            else:
                probs = softmax(scores)
        if scores is None:
            assert len(support) == len(probs)
            scores = np.log(probs)

        self._probs = tuple(probs)
        self._scores = tuple(scores)
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
            return self._scores[self.support.index(e)]
        except ValueError:
            return default

    @property
    def logits(self):
        return self._scores

    def score(self, e):
        try:
            return self._scores[self.support.index(e)]
        except ValueError:
            return -np.inf

    @property
    def scores(self):
        return self._scores

    def sample(self):
        if len(self.support) == 0:
            return
        if len(self.support) == 1:
            return self.support[0]
        return self.support[np.random.choice(len(self.support), p=self._probs)]

    def items(self):
        yield from ((e, p) for e, p in zip(self.support, self.probs) if p > 0.0)

    def keys(self):
        return [e for e in self.support]

    def __len__(self):
        return len(self.support)

    def product(self, other: "DiscreteFactorTable"):
        """
        Product of discrete factor tables

        Combines two factor tables by multiplying their probabilities
        (adding their scores).

        If the support of `self` and `other` are mappings, each (nested) key
        is treated as a variable name. If there are shared variables, these
        are combined by adding their scores when assignment of shared variables
        match. The returned `DiscreteFactorTable` has entries for all
        joint scores greater than -inf combinations of shared variables.

        If the supports of self and other are not mappings, then this behaves
        like the `Multinomial` distribution.
        """

        #Conjunction with null distribution is null table
        if (len(self.support) == 0) or (len(other.support) == 0):
            return DiscreteFactorTable([])

        # NOTE: can this be relaxed?
        assert type(self.support[0]) == type(other.support[0])

        jsupport = []
        jlogits = []
        # HACK: this should throw a warning or something if the distributions have different headers
        # i.e., dictionary keys interact in a weird way
        for si, oi in product(self.support, other.support):
            if isinstance(si, (dict, frozendict)) and isinstance(oi, (dict, frozendict)):
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

    def mix(self, other: "DiscreteFactorTable"):
        """
        Mixture of distributions

        Combines two factor tables by summing their probabilities
        (i.e., taking the logsumexp of their scores)

        If the support of `self` and `other` are mappings, each (nested) key
        is treated as a variable name. If there are shared variables, these
        are combined by adding their scores when assignment of shared variables
        match. The returned `DiscreteFactorTable` has entries for all
        joint scores greater than -inf combinations of shared variables.

        If the supports of self and other are not mappings, then this behaves
        like the `Multinomial` distribution.

        Note that the scores will not be normalized.

        Example:

        pqequalmix = p | q
        pqunequalmix = p*.2 | q*.8
        pqunequalmix = p*.1 | q*.8 | p*.1
        """
        if (len(self.support) == 0):
            return other
        if (len(other.support) == 0):
            return self

        # NOTE: can this be relaxed?
        assert type(self.support[0]) == type(other.support[0])

        jsupport = []
        jlogits = []
        matchedrows = []
        unmatchedrows = []

        #check that all entries have same keys
        if isinstance(self.support[0], (dict, frozendict)):
            s_keys = tuple(self.support[0].keys())
            for si in self.support:
                assert tuple(si.keys()) == s_keys
        if isinstance(other.support[0], (dict, frozendict)):
            o_keys = tuple(other.support[0].keys())
            for oi in self.support:
                assert tuple(oi.keys()) == o_keys

        #first get inner join rows, tracking ones that don't match
        for si, oi in product(self.support, other.support):
            if isinstance(si, (dict, frozendict)) and isinstance(oi, (dict, frozendict)):
                if dict_match(si, oi): #not efficient if the cartesian product is large
                    matchedrows.extend([si, oi])
                    soi = dict_merge(si, oi)
                    if soi in jsupport:
                        continue
                    jprob = np.exp(self.logit(si)) + np.exp(other.logit(oi))
                    jlogit = np.log(jprob)

                    if jlogit == -np.inf:
                        continue
                    jsupport.append(soi)
                    jlogits.append(jlogit)
                else:
                    unmatchedrows.extend([si, oi])
            else:
                soi = (si, oi)
                jprob = np.exp(self.logit(si)) + np.exp(other.logit(oi))
                jlogit = np.log(jprob)
                jsupport.append(soi)
                jlogits.append(jlogit)

        #add in the left and right outer join rows, ensuring that they were never matched
        for i in unmatchedrows:
            if (i in matchedrows) or (i in jsupport):
                continue
            logit = np.log(np.exp(self.logit(i)) + np.exp(other.logit(i)))
            if logit == -np.inf:
                continue
            jsupport.append(i)
            jlogits.append(logit)
        return DiscreteFactorTable(support=jsupport, logits=jlogits)

    def marginalize(self, projection):
        """
        Marginalize a factor table based on a projection

        A projection can be a callable, a python expression as a string,
        a list, or dictionary key.
        """
        mexpscores = DefaultAssignmentMap(lambda _ : 0.0)
        for ele in self.support:
            #try to call it, then try to evaluate it, then try other things
            #could be cleaner...
            try:
                margele = projection(ele)
            except TypeError:
                try:
                    margele = eval(projection, ele)
                except SyntaxError:
                    if isinstance(projection, list):
                        margele = {v: ele[v] for v in projection}
                    else:
                        margele = ele[projection]
            mexpscores[margele] += np.exp(self.score(ele))
        margelements, expscores = zip(*mexpscores.items())
        scores = np.log(expscores)
        return DiscreteFactorTable(support=margelements, scores=scores)

    def __and__(self, other: "DiscreteFactorTable"):
        return self.product(other)

    def __or__(self, other):
        return self.mix(other)

    def __getitem__(self, projection):
        return self.marginalize(projection)

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

    def __repr__(self):
        e_l = ", ".join([f"{e}: {l:.2f}" for e, l in zip(self.support, self.logits)])
        return f"{self.__class__.__name__}({{{e_l}}})"

    def __eq__(self, other):
        return self.support == other.support and self.logits == other.logits

    def isclose(self, other):
        # This implementation avoids comparing the lengths of
        # support since it's possible to have entries that are near-zero,
        # the default value when asking for a probability.
        # Instead, this implementation checks every assigned probability
        # in each, ensuring the value is close in the other distribution.
        for first, second in [(self, other), (other, self)]:
            for s, p in first.items():
                if not np.isclose(p, second.prob(s)):
                    return False
        return True
