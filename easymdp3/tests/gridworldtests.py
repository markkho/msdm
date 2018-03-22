import unittest
from itertools import product

import numpy as np

from easymdp3.domains.gridworld import GridWorld

class GridWorldTestCase(unittest.TestCase):
    def test_simple_deterministic_world(self):
        state_features = [['w', 'c', 'c', 'c', 'c', 'w'],
                          ['w', 'c', 'c', 'c', 'c', 'w'],
                          ['w', 'a', 'a', 'a', 'a', 'y'],
                          ['w', 'c', 'c', 'c', 'c', 'w'],
                          ['w', 'b', 'b', 'b', 'b', 'w'],
                          ['w', 'c', 'c', 'c', 'c', 'w']]
        w = len(state_features[0])
        h = len(state_features)
        state_features = {(x, y): state_features[h - 1 - y][x] for x, y in
                          product(range(w), range(h))}
        absorbing_states = [(5, 3), ]
        feature_rewards = {'a': -2, 'b': 0, 'c': -1, 'y': 1, 'w': 0}
        states = list(state_features.keys())

        params = {
            'width': w,
            'height': h,
            'state_features': state_features,
            'feature_rewards': feature_rewards,
            'absorbing_states': absorbing_states,
            'init_state': (0, 3),
            'include_intermediate_terminal': True
        }

        gw = GridWorld(**params)
        planner = gw.solve(discount_rate=.99)

        true_policy = {(-2, -2): '%',
                       (-1, -1): '%', (0, 0): '^', (0, 1): '>', (0, 2): 'v',
                       (0, 3): 'v', (0, 4): 'v', (0, 5): 'v', (1, 0): '^',
                       (1, 1): '>', (1, 2): 'v', (1, 3): '<', (1, 4): '<',
                       (1, 5): '<', (2, 0): '^', (2, 1): '>', (2, 2): 'v',
                       (2, 3): 'v', (2, 4): '<', (2, 5): '<', (3, 0): '^',
                       (3, 1): '>', (3, 2): 'v', (3, 3): 'v', (3, 4): '>',
                       (3, 5): '>', (4, 0): '>', (4, 1): '>', (4, 2): '>',
                       (4, 3): '>', (4, 4): '>', (4, 5): '>', (5, 0): '^',
                       (5, 1): '^', (5, 2): '^', (5, 3): '%', (5, 4): 'v',
                       (5, 5): 'v'}

        self.assertEqual(planner.optimal_policy, true_policy)

        traj = []
        s = gw.get_init_state()
        while s not in gw.absorbing_states:
            a = planner.optimal_policy[s]
            ns = gw.transition(s, a)
            r = gw.reward(s, a, ns)
            traj.append((s, a, ns, r))
            s = ns
            if len(traj) > 100:
                break
        true_traj = [((0, 3), 'v', (0, 2), 0),
                     ((0, 2), 'v', (0, 1), 0),
                     ((0, 1), '>', (1, 1), 0),
                     ((1, 1), '>', (2, 1), 0),
                     ((2, 1), '>', (3, 1), 0),
                     ((3, 1), '>', (4, 1), 0),
                     ((4, 1), '>', (5, 1), 0),
                     ((5, 1), '^', (5, 2), 0),
                     ((5, 2), '^', (5, 3), 1)]
        self.assertEqual(traj, true_traj)

    def test_slip_state_world(self):
        state_features = [['w', 'x', 'x', 'x', 'x', 'w'],
                          ['w', 'a', 'a', 'a', 'a', 'y'],
                          ['w', 'x', 'x', 'x', 'x', 'w']]
        w = len(state_features[0])
        h = len(state_features)
        state_features = {(x, y): state_features[h - 1 - y][x] for x, y in
                          product(range(w), range(h))}
        absorbing_states = [(5, 1), ]
        feature_rewards = {'a': 0, 'b': 0, 'x': -1, 'y': 5, 'w': 0}

        slip_features = {
            'a': {
                'forward': .6,
                'side': .4,
                'back': 0
            }
        }

        params = {
            'width': w,
            'height': h,
            'state_features': state_features,
            'feature_rewards': feature_rewards,
            'absorbing_states': absorbing_states,
            'slip_features': slip_features,
            'init_state': (0, 1),
            'include_intermediate_terminal': True
        }

        gw = GridWorld(**params)
        planner = gw.solve(discount_rate=.99)



        true_policy = {(-2, -2): '%', (-1, -1): '%',  (0, 0): '^', (0, 1): '>',
                       (0, 2): 'v', (1, 0): '^', (1, 1): '>', (1, 2): 'v',
                       (2, 0): '^', (2, 1): '>', (2, 2): 'v', (3, 0): '>',
                       (3, 1): '>', (3, 2): '>', (4, 0): '>', (4, 1): '>',
                       (4, 2): '>', (5, 0): '^', (5, 1): '%', (5, 2): 'v'}

        self.assertEqual(planner.optimal_policy, true_policy)

        np.random.seed(2223124)
        traj = []
        s = gw.get_init_state()
        for _ in range(20):
            a = planner.optimal_policy[s]
            ns = gw.transition(s, a)
            r = gw.reward(s, a, ns)
            traj.append((s, a, ns, r))
            s = ns
            if s in gw.absorbing_states:
                break
        true_traj = [((0, 1), '>', (1, 1), 0),
                     ((1, 1), '>', (1, 2), -1),
                     ((1, 2), 'v', (1, 1), 0),
                     ((1, 1), '>', (2, 1), 0),
                     ((2, 1), '>', (3, 1), 0),
                     ((3, 1), '>', (4, 1), 0),
                     ((4, 1), '>', (4, 0), -1),
                     ((4, 0), '>', (5, 0), 0),
                     ((5, 0), '^', (5, 1), 5)]
        self.assertEqual(traj, true_traj)

if __name__ == '__main__':
    unittest.main()
