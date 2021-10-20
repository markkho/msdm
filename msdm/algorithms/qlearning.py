from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.distributions import DictDistribution
from collections import defaultdict
from abc import abstractmethod
import random
import math

def epsilon_softmax(action_values, rand_choose, softmax_temp, rng):
    aa, qs = zip(*action_values.items())
    if (rand_choose) and (rng.random() < rand_choose):
        a = rng.choice(aa)
    else:
        if softmax_temp != 0.0:
            weights = [math.exp(qi/softmax_temp) for qi in qs]
            tot = sum(weights)
            weights = [w/tot for w in weights]
            a = rng.choices(aa, weights=weights, k=1)[0]
        else:
            maxq = max(qs)
            a = rng.choice([a for a in aa if action_values[a] == maxq])
    return a

class TemporalDifferenceLearning(Learns):
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

    @abstractmethod
    def _training(self, mdp, rng):
        pass

    def _init_random_number_generator(self):
        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random
        return rng

    def _create_policy(self, mdp, q):
        policy = {}
        for s in mdp.state_list:
            if s not in q:
                max_aa = mdp.actions(s)
            else:
                maxq = max(q[s].values())
                max_aa = [a for a in q[s].keys() if q[s][a] == maxq]
            policy[s] = DictDistribution({a: 1/len(max_aa) for a in max_aa})
        policy = TabularPolicy(policy)
        return policy

    def train_on(self, mdp: TabularMarkovDecisionProcess):
        rng = self._init_random_number_generator()
        q, episode_rewards = self._training(mdp, rng)
        return Result(
            episode_rewards=episode_rewards,
            q_values=q,
            policy=self._create_policy(mdp, q, rng)
        )

class QLearning(TemporalDifferenceLearning):
    r"""
    Q-learning is an off-policy temporal difference control method.
    The temporal difference error in q-learning is:
    $$
    \delta_t = R_{t+1} + \gamma\max_a Q(S_{t+1}, a) - Q(S_t, A_t)
    $$
    """
    def _training(self, mdp, rng):
        q = {}
        initial_q = 0
        episode_rewards = []
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            ep_reward = 0
            while not mdp.is_terminal(s):
                # initialize q values if state hasn't been visited
                if s not in q:
                    q[s] = {a: initial_q for a in mdp.actions(s)}

                # select action
                a = epsilon_softmax(q[s], self.rand_choose, self.softmax_temp, rng)

                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)

                # delta rule
                q[s][a] += self.step_size*(r + mdp.discount_rate*max(q.get(ns, {0: 0}).values()) - q[s][a])

                ep_reward += r
                s = ns
            episode_rewards.append(ep_reward)
        return q, episode_rewards
