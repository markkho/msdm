from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
from collections import defaultdict
import random
import math

class QLearning(Learns):
    def __init__(
        self,
        episodes=100,
        step_size=.1,
        rand_choose=0.05,
        softmax_temp=0.0,
        seed=None
    ):
        """
        Standard Q-learning as described in Sutton & Barto (2018) section 6.5.
        Works on TabularMDPs.

        Parameters:
            episodes - the number of episodes to train
            step_size - the step size of the update rule
            rand_choose - probability of choosing an action uniformly (as in epsilon-greedy)
            softmax_temp - temperature of softmax during learning (closer to 0 -> closer to hardmax)
            seed - seed for random number generator (reset at the beginning of each call to train_on)
        """
        self.episodes = episodes
        self.step_size = step_size
        self.rand_choose = rand_choose
        self.softmax_temp = softmax_temp
        self.seed = seed

    def train_on(self, mdp: TabularMarkovDecisionProcess):
        # initialize random number generator
        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random

        q = {}
        initial_q = 0
        episode_rewards = []
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            ep_reward = 0
            while not mdp.is_terminal(s):
                # initialize q values if state hasn't been visited
                if s not in q:
                    aa = mdp.actions(s)
                    q[s] = {a: initial_q for a in aa}
                # choose according to an epsilon-softmax policy
                if (self.rand_choose) and (rng.random() < self.rand_choose):
                    a = rng.choice(aa)
                else:
                    if self.softmax_temp != 0.0:
                        aa, qs = zip(*q[s].items())
                        weights = [math.exp(qi/self.softmax_temp) for qi in qs]
                        tot = sum(weights)
                        weights = [w/tot for w in weights]
                        a = rng.choices(aa, weights=weights, k=1)[0]
                    else:
                        maxq = max(q[s].values())
                        a = rng.choice([a for a in q[s].keys() if q[s][a] == maxq])

                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)

                # delta rule
                q[s][a] += self.step_size*(r + mdp.discount_rate*max(q.get(ns, {0: 0}).values()) - q[s][a])

                ep_reward += r
                s = ns
            episode_rewards.append(ep_reward)

        policy = {}
        for s in mdp.state_list:
            if s not in q:
                policy[s] = rng.choice(mdp.actions(s))
                continue
            maxq = max(q[s].values())
            policy[s] = rng.choice([a for a in q[s].keys() if q[s][a] == maxq])
        policy = TabularPolicy.from_deterministic_map(policy)

        return Result(
            episode_rewards=episode_rewards,
            q_values=q,
            policy=policy
        )
