import unittest
import numpy as np
from msdm.domains import GridWorld
from msdm.algorithms import VectorizedValueIteration
from msdm.core.problemclasses.mdp import TabularPolicy
from msdm.core.assignment import DefaultAssignmentMap, AssignmentMap

np.seterr(divide='ignore')

class CoreTestCase(unittest.TestCase):
    def test_tabularMDP_ANDcomposition(self):
        gw1 = GridWorld(
            tile_array=[
                '...g',
                '....',
                '.###',
                's..s'
            ],
            step_cost=-1,
            success_prob=.99,
            termination_prob=1e-5
        )

        gw2 = GridWorld(
            tile_array=[
                '##..',
                's..g',
                '....',
                's...'
            ],
            step_cost=0,
            success_prob=.99,
            termination_prob=1e-5
        )
        gw3 = gw1 & gw2

        #test state, aciton, variable composition is consistent
        self.assertTrue(all(s3 == s2 for s3, s2 in zip(gw3.state_list, gw2.state_list)))
        self.assertTrue(all(a3 == a2 for a3, a2 in zip(gw3.action_list, gw2.action_list)))

        #test that mdp distributions are consistent
        s0 = gw1.initial_state_vec * gw2.initial_state_vec
        s0 = s0 / s0.sum()
        eqS0 = np.isclose(gw3.initial_state_vec, s0).all()
        self.assertTrue(eqS0)

        #test that less than or equal are reachable
        rv = sum(gw1.reachable_state_vec * gw2.reachable_state_vec)
        self.assertTrue(sum(gw3.reachable_state_vec) <= rv)

        #test reward composition
        rs = gw3.reachable_state_vec * gw1.reachable_state_vec * gw2.reachable_state_vec
        ast = gw3.absorbing_state_vec * gw1.absorbing_state_vec * gw2.absorbing_state_vec
        ignore = rs[None, None, :]*rs[:, None, None]*ast[:, None, None]*ast[None, None, :]
        eqRF = (ignore * gw3.reward_matrix == ignore * (gw1.reward_matrix + gw2.reward_matrix)).all()
        self.assertTrue(eqRF)

        #test simple transition composition
        tf = (gw1.transition_matrix * gw2.transition_matrix)
        tf = tf / tf.sum(axis=-1, keepdims=True)
        eqTF = np.isclose(gw3.transition_matrix, tf).all()
        self.assertTrue(eqTF)

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
#         self.assertTrue(gw1.is_terminal(stateTraj[-1]))
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