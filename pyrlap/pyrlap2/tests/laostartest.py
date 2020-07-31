import unittest
import numpy as np

from pyrlap.pyrlap2.algorithms.laostar import LAOStar
from pyrlap.pyrlap2.domains import GridWorld
from pyrlap.pyrlap2.core import MarkovDecisionProcess, State


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
            return -np.sum(np.abs(np.array(s[1]) - np.array(goal[1])))

        lao = LAOStar()
        R = lao.planOn(mdp, 
                       heuristic, 
                       maxLAOIters=100,
                       policyEvaluationIters=40,
                       seed=6066253173235511770
                      )
        
if __name__ == '__main__':
    unittest.main()
