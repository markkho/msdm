import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.algorithms import VectorizedValueIteration
from msdm.core.problemclasses.mdp import TabularPolicy

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

    def test_policy_evaluation(self):
        gw1 = GridWorld(
            tile_array=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            step_cost=-1,
        )
        vi = VectorizedValueIteration()
        res = vi.plan_on(gw1)
        res.policy : TabularPolicy
        eval_res = res.policy.evaluate_on(gw1)
        ss0 = gw1.initial_states
        true_v0 = sum([gw1.initial_state_dist().prob(s0)*res.valuefunc[s0] for s0 in ss0])
        print(eval_res.initial_value, true_v0)
        assert eval_res.initial_value == true_v0

if __name__ == '__main__':
    unittest.main()