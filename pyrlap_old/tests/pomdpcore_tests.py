import unittest

import torch

from pyrlap_old.domains.tiger import TigerProblem, TigerCounterAgent
from pyrlap_old.core.pomdp import RandomPOMDPAgent
from pyrlap_old.core.pomdp import MooreFiniteStateController

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

    def test_finite_state_controller(self):
        tp = TigerProblem(
            listen_cost=-1,
            tiger_cost=-100,
            notiger_reward=10,
            roar_prob=.99
        )
        fsc = MooreFiniteStateController(
            tp,
            modes = ["L", "X", "R"],
            initial_mode_dist={"X": 1.0},
            encoder_dist={
                "left-roar": {
                    "X": {"L": 1.0},
                    "L": {"X": 1.0},
                    "R": {"X": 1.0}
                },
                "right-roar": {
                    "X": {"R": 1.0},
                    "L": {"X": 1.0},
                    "R": {"X": 1.0}
                },
                "reset": {
                    "X": {"X": 1.0},
                    "L": {"X": 1.0},
                    "R": {"X": 1.0}
                }
            },
            actor_dist={
                "X": {"listen": 1.0},
                "L": {"right-door": 1.0},
                "R": {"left-door": 1.0}
            }
        )
        reward = [t.r for t in fsc.run(max_steps=1000)]
        mean_reward = sum(reward) / len(reward)
        self.assertTrue(mean_reward > 0)

if __name__ == '__main__':
    unittest.main()
