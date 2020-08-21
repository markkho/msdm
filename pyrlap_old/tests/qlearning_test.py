import unittest
import numpy as np

from pyrlap_old.domains.gridworld import GridWorld
from pyrlap_old.algorithms.qlearning import Qlearning

class QLearningTests(unittest.TestCase):
    def test_qlearner_on_simple_deterministic_gridworld(self):
        gw = GridWorld(
            gridworld_array=['...........',
                             '.xxxxxxxxxy',
                             '.xxxxxxxxxx'],
            absorbing_states=[(10, 1), ],
            init_state=(0, 1),
            feature_rewards={'.': -1, 'x': -10, 'y': 100})

        np.random.seed(123)
        params = {'learning_rate': 1,
                  'eligibility_trace_decay': .8,
                  'initial_qvalue': 100}
        qlearn = Qlearning(gw,
                           softmax_temp=1,
                           discount_rate=.99,
                           **params)
        qlearn.train(episodes=100, max_steps=100)
        test = qlearn.run(softmax_temp=0.0, randchoose=0.0, max_steps=50)
        totr = sum([r for s, a, ns, r in test])

        self.assertEqual(totr, 89)

if __name__ == '__main__':
    unittest.main()
