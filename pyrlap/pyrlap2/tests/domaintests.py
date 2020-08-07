import unittest
import numpy as np
from pyrlap.pyrlap2.domains import GridWorld
from pyrlap.pyrlap2.domains import StickyActionMDP
from pyrlap.pyrlap2.algorithms import VectorizedValueIteration

np.seterr(divide='ignore')

class DomainTestCase(unittest.TestCase):
    def test_stickyactiongridworld(self):
        gw = GridWorld(
            tileArray=[
                'g.......#',
                '.........',
                '.........',
                '.........',
                '........s'
            ], 
            successProb=1.0,
            featureRewards={
                'g': 0,
            },
            stepCost=-1
        )

        sagw = StickyActionMDP(gw, initAction={'dx': 0, 'dy': 0})
        vi = VectorizedValueIteration(
            iterations=50,
            discountRate=1.0,
            entropyRegularization=False,
            temperature=0.0
        )
        vi.planOn(sagw)
        saTraj = vi.policy.runOn(sagw)['stateTraj']
        self.assertTrue(saTraj[0] == {'curAction': {'dx': 0, 'dy': 0}, 'groundState': {'x': 8, 'y': 0}})
        
if __name__ == '__main__':
    unittest.main()


