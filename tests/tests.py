import unittest

from reward_function import RewardFunction

class RewardFunctionTestCase(unittest.TestCase):
    def test_reward_dict_hash_function(self):
        r1 = {'a': 1, 'b': 10, 'c': -100}
        r2 = {'a': 1, 'b': 10, 'c': -100}

        r1 = RewardFunction(reward_dict=r1)
        r2 = RewardFunction(reward_dict=r2)
        self.assertEqual(hash(r1), hash(r2))

    def test_reward_feature_hash_function(self):
        #note: this will not detect hash changes between mypythonlib runs
        f1 = {'a': ('x', 'y', 'z'), 'b': ('x', 'z'), 'c': ()}
        r1 = {'x': 10, 'y': -100, 'z': 4}

        f2 = {'a': ('x', 'y', 'z'), 'b': ('x', 'z'), 'c': ()}
        r2 = {'x': 10, 'y': -100, 'z': 4}

        r1 = RewardFunction(state_features=f1, feature_rewards=r1)
        r2 = RewardFunction(state_features=f2, feature_rewards=r2)
        self.assertEqual(hash(r1), hash(r2))

if __name__ == '__main__':
    unittest.main()
