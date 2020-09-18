import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.domains.gridgame.gridgame import GridGame
from msdm.domains.gridgame.tabulargridgame import TabularGridGame
from msdm.domains import StickyActionMDP
from msdm.algorithms import VectorizedValueIteration

np.seterr(divide='ignore')

class DomainTestCase(unittest.TestCase):
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

        sagw = StickyActionMDP(gw, initAction={'dx': 0, 'dy': 0})
        vi = VectorizedValueIteration(
            iterations=50,
            discount_rate=1.0,
            entropy_regularization=False,
            temperature=0.0
        )
        res = vi.plan_on(sagw)
        saTraj = res.policy.run_on(sagw)['stateTraj']
        self.assertTrue(saTraj[0] == {'curAction': {'dx': 0, 'dy': 0}, 'groundState': {'x': 8, 'y': 0}})

    def test_gridgame_initialization(self):
        #example usage
        gameString = """
            #  # # #  G0 #  # # # 
            G0 . . A0 .  A1 . . G1
            #  # # #  G1 #  # # #
        """.strip()

        gg = GridGame(gameString)
        s = gg.initial_state_dist().sample()
        a = {'A0': {'x': 1, 'y': 0}, 'A1': {'x': -1, 'y': 0}}
        nsdist = gg.next_state_dist(s, a)
        ns = nsdist.sample()
        r = gg.joint_rewards(s, a, ns)
        self.assertTrue(True)
    
    def test_TabularGridGame(self):
        gamestring = """
        # # # # # # # #
        # A0 . . . . A1 #
        # . . . . . . # 
        # u u . . u u #
        # . . . . . . # 
        # G1 . . . G0 . #
        # # # # # # # # 
        """.strip()
        gg = TabularGridGame(gamestring)
        init_state = gg.initial_state_dist().sample()
        action = gg.joint_action_dist(init_state).sample()
        next_state = gg.next_state_dist(init_state,action).sample()
        rewards = gg.joint_rewards(init_state,action,next_state)
        
if __name__ == '__main__':
    unittest.main()

