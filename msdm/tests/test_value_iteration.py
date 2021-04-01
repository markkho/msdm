import unittest

import numpy as np
from frozendict import frozendict
from msdm.core.distributions import DictDistribution
from msdm.algorithms import ValueIteration
from msdm.tests.domains import Counter, GNTFig6_6, Geometric, VaryingActionNumber
from msdm.domains import GridWorld

class VITestCase(unittest.TestCase):
    def test_value_iteration(self):
        mdp = Counter(3)
        res = ValueIteration().plan_on(mdp)
        out = res.policy.run_on(mdp)
        assert out.state_traj == (0, 1, 2)
        assert out.action_traj == (1, 1, 1)
        assert res.policy.action(0) == 1
        assert res.policy.action(1) == 1
        assert res.policy.action(2) == 1

    def test_value_iteration_geometric(self):
        mdp = Geometric(p=1/13)
        res = ValueIteration(iterations=500).plan_on(mdp)
        assert np.isclose(res.V[0], -13), res.V

    def test_value_iteration_varying_action_number(self):
        mdp = VaryingActionNumber()
        res = ValueIteration().plan_on(mdp)
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
        res = ValueIteration().plan_on(mdp)
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
