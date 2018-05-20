import unittest
from itertools import product
import sys
import pyximport; pyximport.install()
import logging

from mdp_lib import GridWorld, RewardFunction, Policy
from mdp_lib.util import calc_softmax_policy

# from fsss import ForwardSearchSparseSampling
from fsss_cy import ForwardSearchSparseSampling
from valueheuristic import ValueHeuristic

def setup_gw(tile_features, feature_rewards, absorbing_states, init_state):
    w = len(tile_features[0])
    h = len(tile_features)
    state_features = {}
    for x, y in product(range(w), range(h)):
        state_features[(x, y)] = tile_features[h - 1 - y][x]

    params = {
        'width': w,
        'height': h,
        'state_features': state_features,
        'feature_rewards': feature_rewards,
        'absorbing_states': absorbing_states,
        'init_state': init_state
    }

    gw = GridWorld(**params)
    return gw

class FSSSTestCase(unittest.TestCase):
    def test_find_long_path_to_goal(self):
        tiles = [['w', 'w', 'w', 'x', 'w', 'w', 'w'],
                 ['w', 'x', 'w', 'w', 'w', 'x', 'y']]
        absorbing_states = [(6, 0), ]
        feature_rewards = {
            'w': 0,
            'y': 1,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=True)
        fsss.search(init_state=gw.get_init_state(),
                    max_iterations=1000)
        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '^', 0), ((0, 1), '>', 0),
                     ((1, 1), '>', 0), ((2, 1), 'v', 0),
                     ((2, 0), '>', 0), ((3, 0), '>', 0),
                     ((4, 0), '^', 0), ((4, 1), '>', 0),
                     ((5, 1), '>', 0), ((6, 1), 'v', 1),
                     ((6, 0), '%', 0)]
        self.assertEqual(traj, true_traj)

    def test_single_step_to_goal(self):
        tiles = [['x', 'w'],
                 ['w', 'y']]
        absorbing_states = [(1, 0), (0, 1), (1, 1)]
        feature_rewards = {
            'w': 0,
            'y': 1,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=0.1,
                                           value_error=.01,
                                           break_ties_randomly=True)
        fsss.search(init_state=gw.get_init_state())
        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '>', 1), ((1, 0), '%', 0)]
        self.assertEqual(traj, true_traj)

    def test_find_path_through_open_space(self):
        tiles = [['w', 'w', 'w', 'w'],
                 ['w', 'w', 'x', 'w'],
                 ['w', 'w', 'x', 'y'],
                 ['w', 'w', 'x', 'w'],
                 ['w', 'w', 'x', 'w']]
        absorbing_states = [(3, 2), ]
        feature_rewards = {
            'w': 0,
            'y': 1,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=False)
        fsss.search(init_state=gw.get_init_state())
        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '^', 0), ((0, 1), '^', 0),
                     ((0, 2), '^', 0), ((0, 3), '^', 0),
                     ((0, 4), '>', 0), ((1, 4), '>', 0),
                     ((2, 4), '>', 0), ((3, 4), 'v', 0),
                     ((3, 3), 'v', 1), ((3, 2), '%', 0)]
        self.assertEqual(traj, true_traj)

    def test_find_path_through_open_space_with_action_decoupling(self):
        tiles = [['w', 'w', 'w', 'w'],
                 ['w', 'w', 'x', 'w'],
                 ['w', 'w', 'x', 'y'],
                 ['w', 'w', 'x', 'w'],
                 ['w', 'w', 'x', 'w']]
        absorbing_states = [(3, 2), ]
        feature_rewards = {
            'w': 0,
            'y': 1,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=False)
        fsss.search(init_state=gw.get_init_state(),
                    termination_condition="decoupled_values")
        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '^', 0), ((0, 1), '^', 0),
                     ((0, 2), '^', 0), ((0, 3), '^', 0),
                     ((0, 4), '>', 0), ((1, 4), '>', 0),
                     ((2, 4), '>', 0), ((3, 4), 'v', 0),
                     ((3, 3), 'v', 1), ((3, 2), '%', 0)]
        self.assertEqual(traj, true_traj)

    def test_circular_path(self):
        gw = GridWorld(width=2, height=2,
                       reward_dict={
                           (0, 0): {'>': 1},
                           (1, 0): {'^': 1},
                           (1, 1): {'<': 1},
                           (0, 1): {'v': 1}
                       },
                       default_reward=-1,
                       init_state=(0, 0))

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=True)
        fsss.search(init_state=gw.get_init_state())
        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=10)
        true_traj = [((0, 0), '>', 1), ((1, 0), '^', 1),
                     ((1, 1), '<', 1), ((0, 1), 'v', 1),
                     ((0, 0), '>', 1), ((1, 0), '^', 1),
                     ((1, 1), '<', 1), ((0, 1), 'v', 1),
                     ((0, 0), '>', 1), ((1, 0), '^', 1)]
        self.assertEqual(traj, true_traj)

    def test_value_heuristic(self):
        tiles = [['w', 'w', 'w', 'x', 'w', 'w', 'w'],
                 ['w', 'x', 'w', 'w', 'w', 'x', 'y']]
        absorbing_states = [(6, 0), ]
        feature_rewards = {
            'w': 0,
            'y': 5,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        vh = ValueHeuristic(5, -1, .95, vmax=5)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=True,
                                           valueheuristic=vh)
        fsss.search(init_state=gw.get_init_state(),
                    termination_condition="converged_values",
                    value_error=.01,
                    max_iterations=1000)

        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '^', 0), ((0, 1), '>', 0),
                     ((1, 1), '>', 0), ((2, 1), 'v', 0),
                     ((2, 0), '>', 0), ((3, 0), '>', 0),
                     ((4, 0), '^', 0), ((4, 1), '>', 0),
                     ((5, 1), '>', 0), ((6, 1), 'v', 5),
                     ((6, 0), '%', 0)]
        self.assertEqual(traj, true_traj)


    def test_expansion_policy(self):
        tiles = [['w', 'w', 'w', 'x', 'w', 'w', 'w'],
                 ['w', 'x', 'w', 'w', 'w', 'x', 'y']]
        absorbing_states = [(6, 0), ]
        feature_rewards = {
            'w': 0,
            'y': 5,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)
        gw.solve(gamma=.95)
        policy = calc_softmax_policy(gw.action_value_function, temp=.02)
        policy = Policy(policy)

        vh = ValueHeuristic(5, -1, .95, vmax=5)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=True,
                                           valueheuristic=vh,
                                           expansion_policy=policy)
        fsss.search(init_state=gw.get_init_state(),
                    termination_condition="converged_values",
                    value_error=.01,
                    max_iterations=1000,
                    initial_action_rule='expansion_policy')

        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        true_traj = [((0, 0), '^', 0), ((0, 1), '>', 0),
                     ((1, 1), '>', 0), ((2, 1), 'v', 0),
                     ((2, 0), '>', 0), ((3, 0), '>', 0),
                     ((4, 0), '^', 0), ((4, 1), '>', 0),
                     ((5, 1), '>', 0), ((6, 1), 'v', 5),
                     ((6, 0), '%', 0)]
        self.assertEqual(traj, true_traj)

    def test_clearing_unused_nodes(self):
        tiles = [['w', 'w', 'w', 'x', 'w', 'w', 'w'],
                 ['w', 'x', 'w', 'w', 'w', 'x', 'y']]
        absorbing_states = [(6, 0), ]
        feature_rewards = {
            'w': 0,
            'y': 1,
            'x': -1
        }
        init_state = (0, 0)
        gw = setup_gw(tiles, feature_rewards, absorbing_states, init_state)

        fsss = ForwardSearchSparseSampling(gw,
                                           discount_rate=.95,
                                           value_error=.01,
                                           break_ties_randomly=True,
                                           max_depth=100)
        fsss.search(init_state=gw.get_init_state(),
                    termination_condition="converged_values")

        pre_cleaning_visits = fsss.size()

        traj = fsss.unroll(start=gw.get_init_state(),
                           steps=100)
        traj = [(s, a) for s, a, r in traj]
        fsss.set_root_search_trajectory(traj)
        fsss.clear_solved_nodes()

        post_cleaning_visits = fsss.size()

        fsss.search(init_state=gw.get_init_state(),
                    termination_condition="converged_values",
                    max_iterations=1)

        post_research_visits = fsss.size()
        print (pre_cleaning_visits, post_cleaning_visits, post_research_visits)
        self.assertTrue(pre_cleaning_visits > post_research_visits)
        self.assertTrue(post_research_visits >= post_cleaning_visits)

# logging.basicConfig(stream=sys.stdout,
#                         level=logging.DEBUG,
#                         format='%(asctime)s : %(name)s : %(message)s',
#                         datefmt='%H:%M:%S')

if __name__ == '__main__':
    unittest.main()
