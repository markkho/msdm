import unittest

from pyrlap.domains.tiger import TigerProblem, TigerCounterAgent
from pyrlap.core.pomdp import RandomPOMDPAgent

class POMDPCoreTests(unittest.TestCase):
    def test_counter_agent(self):
        tp = TigerProblem(
            listen_cost=-1,
            tiger_cost=-100,
            notiger_reward=10,
            roar_prob=.95
        )
        tca = TigerCounterAgent(tp, count_diff=2)
        reward = [t.r for t in tca.run(max_steps=1000)]
        mean_reward = sum(reward)/len(reward)
        self.assertTrue(mean_reward > 0)

    def test_random_agent(self):
        tp = TigerProblem(
            listen_cost=-1,
            tiger_cost=-100,
            notiger_reward=10,
            roar_prob=.95
        )
        ra = RandomPOMDPAgent(tp)
        reward = [t.r for t in ra.run(max_steps=1000)]
        mean_reward = sum(reward) / len(reward)
        self.assertTrue(mean_reward < 0)

if __name__ == '__main__':
    unittest.main()
