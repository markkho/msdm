import unittest
import numpy as np
from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.algorithms import ValueIteration
from msdm.core.problemclasses.mdp import TabularPolicy, TabularMarkovDecisionProcess

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

    def test_tabular_mdp(self):
        from itertools import product
        from msdm.algorithms import PolicyIteration, ValueIteration
        from msdm.tests.domains import make_russell_norvig_grid

        for dr, sp in product([.99, .9, .5, .1], [.1, .2, .5, .8, 1.0]):
            # Create MDPs and copy them
            g1 = make_russell_norvig_grid(
                discount_rate=dr,
                slip_prob=sp
            )
            g2 = TabularMarkovDecisionProcess.from_matrices(
                state_list = g1.state_list,
                action_list = g1.action_list,
                initial_state_vec = g1.initial_state_vec,
                transition_matrix = g1.transition_matrix,
                reward_matrix = g1.reward_matrix,
                nonterminal_state_vec = g1.nonterminal_state_vec,
                discount_rate = g1.discount_rate
            )
            pi1 = PolicyIteration().plan_on(g1)
            pi2 = PolicyIteration().plan_on(g2)

            assert g2.state_list == g1.state_list
            assert id(g2.state_list) != id(g1.state_list)
            assert g2.action_list == g1.action_list
            assert id(g2.action_list) != id(g1.action_list)
            assert (g2.initial_state_vec == g1.initial_state_vec).all()
            assert id(g2.initial_state_vec) != id(g1.initial_state_vec)
            assert (g2.transition_matrix == g1.transition_matrix).all()
            assert id(g2.transition_matrix) != id(g1.transition_matrix)
            assert (g2.reward_matrix == g1.reward_matrix).all()
            assert id(g2.reward_matrix) != id(g1.reward_matrix)
            assert (g2.nonterminal_state_vec == g1.nonterminal_state_vec).all()
            assert id(g2.nonterminal_state_vec) != id(g1.nonterminal_state_vec)
            assert g2.discount_rate == g1.discount_rate

            assert pi1.initial_value == pi2.initial_value
            assert pi1.iterations == pi2.iterations
            assert pi1.converged == pi2.converged

if __name__ == '__main__':
    unittest.main()
