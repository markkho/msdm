import unittest
import numpy as np

from pyrlap.pyrlap2.algorithms.laostar import LAOStar
from pyrlap.pyrlap2.domains import GridWorld


np.seterr(divide='ignore')

class LAOStarTestCase(unittest.TestCase):
    def test_DeterministicLAOStarGridWorld(self):
        gw = GridWorld(
            tileArray=[
                '......g',
                '...####',
                '.###...',
                '.....##',
                '..####.',
                '..s....',
            ],
            featureRewards={'g': 0},
            stepCost=-1,
            terminationProb=.0
        )
        mdp = gw

        goal = mdp.absorbingStates[0]
        def heuristic(s):
            if mdp.isTerminal(s):
                return 0.0
            return -np.sum(np.abs(np.array(s['x']) - np.array(goal['x'])))

        lao = LAOStar(
            heuristic,
            maxLAOIters=100,
            policyEvaluationIters=40,
            seed=6066253173235511770
        )
        R = lao.planOn(mdp)
        
if __name__ == '__main__':
    unittest.main()
