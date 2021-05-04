import unittest
import numpy as np
from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.algorithms import ValueIteration
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
        vi = ValueIteration()
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
        vi = ValueIteration()
        res = vi.plan_on(gw1)
        res.policy : TabularPolicy
        eval_res = res.policy.evaluate_on(gw1)
        ss0 = gw1.initial_states
        true_v0 = sum([gw1.initial_state_dist().prob(s0)*res.valuefunc[s0] for s0 in ss0])
        assert eval_res.initial_value == true_v0
        assert eval_res.occupancy[frozendict({'x': 0, 'y': 1})] == 1

    def test_quick_tabular_mdp(self):
        from msdm.core.distributions import DictDistribution as DD
        from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP
        TERMINAL = -float('inf')
        for mdp in [
            QuickTabularMDP(
                next_state_dist    = lambda s, a : DD({s + a: .8, s - a: .1, s: .1 if a else 1}) if (0 < s < 10) else DD({TERMINAL: 1}),
                reward             = lambda s, a, ns : {0: -100, 10: 0, TERMINAL: 0}.get(ns, 0) - abs(a) - 1,
                actions            = lambda s: (-1, 0, 1),
                initial_state_dist = lambda : DD({5: .5, 6: .5}),
                is_terminal        = lambda s: s == TERMINAL,
                discount_rate      = .99
            ),
            QuickTabularMDP(
                next_state         = lambda s, a : (s + a) if (0 < s < 10) else s,
                reward             = -1,
                actions            = (-1, 0, 1),
                initial_state      = 1,
                is_terminal        = lambda s: s == 10,
                discount_rate      = .99
            ),
        ]:
            res = ValueIteration().plan_on(mdp)
            pi = [res.policy.action_dist(s).sample() for s in range(1, 10)]
            assert all([a == 1 for a in pi])

if __name__ == '__main__':
    unittest.main()
