import unittest
from pyrlap.domains.gridworld import GridWorld

class AgentTestCase(unittest.TestCase):
    def test_successor_rep(self):
        mdp = GridWorld(
            gridworld_array=[
                '...g',
                '.xxx',
                '.x..',
                '....'
            ],
            absorbing_features=['g', ],
            feature_rewards={'g': 100, 'x': -10},
            wall_action=True,
            init_state=(0, 0)
        )
        pol = mdp.solve(.99, softmax_temp=.5)
        occ = pol.calc_occupancy()
        self.assertTrue(occ[(0, 0)] > occ[(1, 0)])
        self.assertTrue(occ[(1, 0)] > occ[(2, 0)])

    def test_value_calculation(self):
        mdp = GridWorld(
            gridworld_array=[
                '...g',
                '.xxx',
                '.x..',
                '....'
            ],
            absorbing_features=['g', ],
            feature_rewards={'g': 100, 'x': -10},
            wall_action=True,
            init_state=(0, 0)
        )
        pol = mdp.solve(.99, softmax_temp=.5)
        vf = pol.value()
        self.assertTrue(vf[(0, 0)] > vf[(1, 0)])
        self.assertTrue(vf[(1, 0)] > vf[(2, 0)])


if __name__ == '__main__':
    unittest.main()
