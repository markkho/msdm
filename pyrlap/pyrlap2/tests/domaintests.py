import unittest
import numpy as np
from pyrlap.pyrlap2.domains import GridWorld
from pyrlap.pyrlap2.domains.gridgame.gridgame import GridGame
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
        res = vi.planOn(sagw)
        saTraj = res.policy.runOn(sagw)['stateTraj']
        self.assertTrue(saTraj[0] == {'curAction': {'dx': 0, 'dy': 0}, 'groundState': {'x': 8, 'y': 0}})

    def test_gridgame_initialization(self):
        #example usage
        gameString = """
            #  # # #  G0 #  # # # 
            G0 . . A0 .  A1 . . G1
            #  # # #  G1 #  # # #
        """.strip()

        gg = GridGame(gameString)
        s = gg.getInitialStateDist().sample()
        a = {'A0': {'x': 1, 'y': 0}, 'A1': {'x': -1, 'y': 0}}
        nsdist = gg.getNextStateDist(s, a)
        ns = nsdist.sample()
        r = gg.getJointRewards(s, a, ns)
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()

