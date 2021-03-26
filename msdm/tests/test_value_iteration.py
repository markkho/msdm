import unittest

import numpy as np
from msdm.algorithms import VectorizedValueIteration
from msdm.tests.domains import Counter, GNTFig6_6, Geometric

class VITestCase(unittest.TestCase):
    def test_value_iteration(self):
        mdp = Counter(3)
        res = VectorizedValueIteration().plan_on(mdp)
        out = res.policy.run_on(mdp)
        assert out.state_traj == (0, 1, 2)
        assert out.action_traj == (1, 1, 1)
        assert res.policy.action(0) == 1
        assert res.policy.action(1) == 1
        assert res.policy.action(2) == 1

    def test_value_iteration_geometric(self):
        mdp = Geometric(p=1/13)
        res = VectorizedValueIteration(iterations=500).plan_on(mdp)
        print(res.policy.action_dist(0))
        assert np.isclose(res.V[0], -13), res.V
