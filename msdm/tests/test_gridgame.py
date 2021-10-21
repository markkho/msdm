import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.domains.gridgame.tabulargridgame import TabularGridGame

np.seterr(divide='ignore')

class TestCase(unittest.TestCase):
    def test_gridgame_initialization(self):
        #example usage
        gameString = """
            #  # # #  G0 #  # # #
            G0 . . A0 .  A1 . . G1
            #  # # #  G1 #  # # #
        """.strip()

        gg = TabularGridGame(gameString)
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
        joint_actions = gg.joint_actions(init_state)
        joint_action = {agent: next(actions) for agent, actions in joint_actions.items()}
        next_state = gg.next_state_dist(init_state,joint_action).sample()
        rewards = gg.joint_rewards(init_state,joint_action,next_state)

if __name__ == '__main__':
    unittest.main()
