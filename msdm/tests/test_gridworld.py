import unittest
import numpy as np
from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.domains.gridgame.tabulargridgame import TabularGridGame
from msdm.domains import StickyActionMDP
from msdm.algorithms import VectorizedValueIteration

np.seterr(divide='ignore')


class TestCase(unittest.TestCase):
    def test_stickyactiongridworld(self):
        gw = GridWorld(
            tile_array=[
                'g.......#',
                '.........',
                '.........',
                '.........',
                '........s'
            ],
            success_prob=1.0,
            feature_rewards={
                'g': 0,
            },
            step_cost=-1
        )

        sagw = StickyActionMDP(gw, init_action=frozendict({'dx': 0, 'dy': 0}))
        vi = VectorizedValueIteration(
            iterations=50,
            discount_rate=1.0,
            entropy_regularization=False,
            temperature=0.0
        )
        res = vi.plan_on(sagw)
        saTraj = res.policy.run_on(sagw).state_traj
        self.assertTrue(saTraj[0] == {'curAction': {'dx': 0, 'dy': 0},
                                      'groundState': {'x': 8, 'y': 0}})

if __name__ == '__main__':
    unittest.main()
