import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.algorithms import VectorizedValueIteration

np.seterr(divide='ignore')

class CoreTestCase(unittest.TestCase):

    def test_runningAgentOnMDP(self):
        gw1 = GridWorld(
            tile_array=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            step_cost=-1,
        )
        vi = VectorizedValueIteration(temperature=.1,
                                      entropy_regularization=True)
        res = vi.plan_on(gw1)
        stateTraj = res.policy.run_on(gw1).state_traj
        self.assertTrue(stateTraj[-1] in gw1.absorbing_states)
        self.assertTrue(stateTraj[0] in gw1.initial_states)

if __name__ == '__main__':
    unittest.main()