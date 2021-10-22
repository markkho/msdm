import unittest
import numpy as np
from frozendict import frozendict
from msdm.core.distributions import DictDistribution
from msdm.algorithms import ValueIteration, PolicyIteration, LRTDP
from msdm.tests.domains import Counter, GNTFig6_6, Geometric, VaryingActionNumber, make_russell_norvig_grid
from msdm.domains import GridWorld


class MyTestCase(unittest.TestCase):
    def test_policy_iteration(self):
        mdp = Counter(3)
        res = PolicyIteration().plan_on(mdp)
        out = res.policy.run_on(mdp)
        assert out.state_traj == (0, 1, 2)
        assert out.action_traj == (1, 1, 1)
        assert res.policy.action(0) == 1
        assert res.policy.action(1) == 1
        assert res.policy.action(2) == 1

    def test_policy_iteration_geometric(self):
        mdp = Geometric(p=1/13)
        res = PolicyIteration(iterations=500).plan_on(mdp)
        assert np.isclose(res.V[0], -13), res.V

    def test_policy_iteration_varying_action_number(self):
        mdp = VaryingActionNumber()
        res = PolicyIteration().plan_on(mdp)
        assert np.isclose(res.V[0], -2), res.V
        assert res.policy.run_on(mdp).action_traj == (+1, +1)

    def test_equal_value(self):
        '''
        In this MDP, the value at the non-initial, non-terminal corners is equal.
        This means the policy at the start state should assign equal probability
        to either.
        '''
        mdp = GridWorld(
            tile_array=[
                '.g',
                's.',
            ],
            feature_rewards={'g': 0},
            step_cost=-1,
        )
        res = PolicyIteration().plan_on(mdp)
        assert np.isclose(res.V[frozendict(x=0, y=1)], res.V[frozendict(x=1, y=0)])
        assert res.policy.action_dist(frozendict(x=0, y=0)).\
            isclose(DictDistribution({
                frozendict({'dx': 0, 'dy': 0}): 0,
                frozendict({'dx': 1, 'dy': 0}): 1/2,
                frozendict({'dx': -1, 'dy': 0}): 0,
                frozendict({'dy': 1, 'dx': 0}): 1/2,
                frozendict({'dy': -1, 'dx': 0}): 0
        }))
        assert res.policy.action_dist(frozendict(x=0, y=1)).isclose(DictDistribution({
                frozendict({'dx': 1, 'dy': 0}): 1,
        }))

    def test_policy_iteration_gridworld(self):
        gw = GridWorld(
            tile_array=[
                '......g',
                '...####',
                '..##...',
                '..#....',
                '.......',
                '####...',
                's......',
            ])
        pi_res = PolicyIteration()(gw)
        vi_res = ValueIteration()(gw)
        lrtdp = LRTDP()(gw)
        assert pi_res.initial_value == vi_res.initial_value == lrtdp.initial_value

    def test_policy_iteration_gridworld2(self):
        gw = GridWorld((
            '..g..',
            '.###.',
            '..#..',
            '..s..'
        ), discount_rate=1 - 1e-5)
        pi = PolicyIteration().plan_on(gw)
        vi = ValueIteration().plan_on(gw)
        reachable = sorted(gw.reachable_states(),
                           key=lambda s: (s['x'], s['y']))
        pi_mat = pi.policy.as_matrix(reachable, gw.action_list)
        vi_mat = vi.policy.as_matrix(reachable, gw.action_list)
        assert (pi_mat == vi_mat).all()
        assert all([np.isclose(pi.valuefunc[s], vi.valuefunc[s])
                    for s in reachable])

    def test_policy_iteration_and_value_iteration_russell_norvig(self):
        for discount_rate in [i/10 for i in range(1, 10)] + [.95, .99, 1.0]:
            for slip_prob in [i/10 for i in range(1, 10)] + [.95, .99, 1.0]:
                gw = make_russell_norvig_grid(
                        discount_rate=discount_rate,
                        slip_prob=slip_prob,
                )
                vi_res = ValueIteration(iterations=int(1e3)).plan_on(gw)
                pi_res = PolicyIteration(iterations=int(1e3)).plan_on(gw)
                assert np.isclose(vi_res._qvaluemat, pi_res._qvaluemat, atol=5e-4).all()

    def test_policy_iteration_heavenorhell(self):
        # technically a pomdp, but we can solve underlying mdp
        from msdm.domains.heavenorhell import HeavenOrHell
        for discount_rate in [i/10 for i in range(1, 10, 2)] + [.95, .99, .99999]:
            for coherence in [i/10 for i in range(1, 10, 2)] + [.95, .99, .99999]:
                print(discount_rate, coherence)

                hh = HeavenOrHell(
                    coherence=coherence,
                    grid=
                        """
                        hcg
                        #.#
                        #s#
                        """,
                    discount_rate=discount_rate,
                    heaven_reward=50,
                    hell_reward=-50,
                )
                pi = PolicyIteration().plan_on(hh)
                vi = ValueIteration().plan_on(hh)
                reachable = sorted(hh.reachable_states())
                pi_mat = pi.policy.as_matrix(reachable, hh.action_list)
                vi_mat = vi.policy.as_matrix(reachable, hh.action_list)
                assert (pi_mat == vi_mat).all()
                assert all([np.isclose(pi.valuefunc[s], vi.valuefunc[s])
                            for s in reachable])

if __name__ == '__main__':
    unittest.main()
