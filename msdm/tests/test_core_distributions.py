import unittest
import numpy as np

import pandas as pd

from msdm.core.distributions import DiscreteFactorTable as Pr

def toDF(p):
    df = pd.DataFrame(p.support)
    df['prob'] = p.probs
    df['scores'] = p.scores
    return df

np.seterr(divide='ignore')

class LAOStarTestCase(unittest.TestCase):
    def test_independent_conjunction(self):
        # Conjunction (independent variables)
        pA = Pr([{'a': 0}, {'a': 1}], probs=[.9, .1])
        pB = Pr([{'b': 0}, {'b': 1}], probs=[.5, .5])
        p = pA & pB
        assert all(np.isclose(
            toDF(p).sort_values(['a', 'b'])['prob'],
            [.45, .45, .05, .05]
        ))

    def test_overlapping_conjunction(self):
        # Conjunction
        pA1 = Pr([{'a': 0}, {'a': 1}, {'a': 2}], probs=[.7, .2, .1])
        pA2 = Pr([{'a': 1}, {'a': 2}, {'a': 3}], probs=[.5, .4, .1])
        p = pA1 & pA2
        pp = np.array([.2 * .5, .1 * .4])
        pp = pp / pp.sum()
        assert all(np.isclose(
            toDF(p.normalize()).sort_values(['a'])['prob'],
            pp)
        )

    def test_dependent_conjunction(self):
        # Conjunction (dependent variables)
        pAB = Pr([{'a': 0, 'b': 0}, {'a': 1, 'b': 1}], probs=[.9, .1])
        pB = Pr([{'b': 0}, {'b': 1}], probs=[1 / 3, 2 / 3])
        p = pAB & pB
        pp = np.array([.9 * (1 / 3), .1 * (2 / 3)])
        pp = pp / pp.sum()
        assert all(np.isclose(
            toDF(p.normalize()).sort_values(['a', 'b'])['prob'],
            pp
        ))

    def test_dependent_conjunction2(self):
        # Conjunction (dependent variables)
        pAB = Pr([{'a': 0, 'b': 0}, {'a': 1, 'b': 1}], probs=[.9, .1])
        pBC = Pr([{'b': 0, 'c': 0}, {'b': 1, 'c': 0}, {'b': 1, 'c': 1}],
                 probs=[1 / 3, 1 / 3, 1 / 3])
        p = pAB & pBC
        pp = np.array([.9 * (1 / 3), .1 * (1 / 3), .1 * (1 / 3)])
        pp = pp / pp.sum()
        assert all(np.isclose(
            toDF(p.normalize()).sort_values(['a', 'b'])['prob'],
            pp
        ))

    def test_disjunction(self):
        # Mixture of Distributions
        pA1 = Pr([{'a': 0}, {'a': 1}], probs=[.9, .1])
        pA2 = Pr([{'a': 1}, {'a': 2}], probs=[.5, .5])
        p = pA1 * .1 | pA2 * .9
        pp = np.array([.9 * .1, (.1 * .1 + .5 * .9), .5 * .9])
        assert all(np.isclose(
            toDF(p.normalize()).sort_values(['a'])['prob'],
            pp
        ))

    def test_marginalization(self):
        # marginalization
        pA = Pr([{'a': 0}, {'a': 1}], probs=[.9, .1])
        pB = Pr([{'b': 0}, {'b': 1}], probs=[.1, .9])
        pAB = pA & pB
        pAeqB = pAB[lambda r: "a == b" if r['a'] == r['b'] else "a != b"]
        pp = np.array([.9 * .1 + .9 * .1, 1 - (.9 * .1 + .9 * .1)])
        assert all(np.isclose(
            toDF(pAeqB)['prob'],
            pp
        ))

if __name__ == '__main__':
    unittest.main()
