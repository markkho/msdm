from msdm.core.algorithmclasses import Learns, Result
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy
from msdm.core.distributions import DictDistribution, SoftmaxDistribution
from msdm.core.utils.dictutils import defaultdict2
from collections import defaultdict
from types import SimpleNamespace
from abc import abstractmethod, ABC
import random
import math

def epsilon_softmax_sample(action_values, rand_choose, softmax_temp, rng):
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

def epsilon_softmax_dist(action_values, rand_choose, softmax_temp):
    if softmax_temp == 0.0:
        maxq = max(action_values.values())
        sm_dist = DictDistribution.uniform([a for a, q in action_values.items() if q == maxq])
    else:
        sm_dist = SoftmaxDistribution({a: q/softmax_temp for a, q in action_values.items()})
    if rand_choose == 0.0:
        return sm_dist
    rand_dist = DictDistribution.uniform(action_values.keys())
    return rand_dist*rand_choose | sm_dist*(1 - rand_choose)

class TDLearningEventListener(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def end_of_timestep(self, local_vars):
        pass
    @abstractmethod
    def end_of_episode(self, local_vars):
        pass
    @abstractmethod
    def results(self):
        pass

class EpisodeRewardEventListener(TDLearningEventListener):
    def __init__(self):
        self.episode_rewards = []
        self.curr_ep_rewards = 0
    def end_of_timestep(self, local_vars):
        self.curr_ep_rewards += local_vars['r']
    def end_of_episode(self, local_vars):
        self.episode_rewards.append(self.curr_ep_rewards)
        self.curr_ep_rewards = 0
    def results(self):
        return SimpleNamespace(
            episode_rewards=self.episode_rewards
        )

class TemporalDifferenceLearning(Learns):
    def __init__(
        self,
        episodes=100,
        step_size=.1,
        rand_choose=0.05,
        softmax_temp=0.0,
        initial_q=0.0,
        seed=None,
        event_listener_type : TDLearningEventListener = EpisodeRewardEventListener
    ):
        """
        Generic temporal difference learning interface.

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
        if isinstance(initial_q, float):
            self.initial_q = lambda s, a: initial_q
        self.event_listener_type = event_listener_type

    @abstractmethod
    def _training(self, mdp, rng):
        """This is the main training loop. It should return
        a nested dictionary. Specifically, a dictionary with
        states as keys and action-value dictionaries as values."""
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
        event_listener = self.event_listener_type()
        q = self._training(mdp, rng, event_listener)
        return Result(
            q_values=q,
            policy=self._create_policy(mdp, q),
            event_listener_results=event_listener.results()
        )

class QLearning(TemporalDifferenceLearning):
    r"""
    Q-learning is an off-policy temporal difference control method.
    The temporal difference error in q-learning is:
    $$
    \delta_t = R_{t+1} + \gamma\max_a Q(S_{t+1}, a) - Q(S_t, A_t)
    $$
    """
    def _training(self, mdp, rng, event_listener):
        initial_avals = lambda s: {a: self.initial_q(s, a) for a in mdp.actions(s)}
        q = defaultdict2(initial_avals, initialize_defaults=True)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_terminal(s):
                # select action
                a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                # update
                q[s][a] += self.step_size*(r + mdp.discount_rate*max(q.get(ns, {0: 0}).values()) - q[s][a])
                # end of timestep
                event_listener.end_of_timestep(locals())
                s = ns
            event_listener.end_of_episode(locals())
        return q

class SARSA(TemporalDifferenceLearning):
    r"""
    SARSA is an on-policy temporal difference control method.
    The temporal difference error in SARSA is:
    $$
    \delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    $$
    """
    def _training(self, mdp, rng, event_listener):
        initial_avals = lambda s: {a: self.initial_q(s, a) for a in mdp.actions(s)}
        q = defaultdict2(initial_avals, initialize_defaults=True)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            if s not in q:
                q[s] = {a: self.initial_q(s, a) for a in mdp.actions(s)}
            a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
            while not mdp.is_terminal(s):
                # get next state, reward, next action
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                na = epsilon_softmax_sample(q[ns], self.rand_choose, self.softmax_temp, rng)
                # update
                q[s][a] += self.step_size*(r + mdp.discount_rate*q[ns][na] - q[s][a])
                # end of timestep
                event_listener.end_of_timestep(locals())
                s, a = ns, na
            event_listener.end_of_episode(locals())
        return q

class ExpectedSARSA(TemporalDifferenceLearning):
    r"""
    Expected SARSA is an on-policy temporal difference control method.
    The temporal difference error in Expected SARSA is:
    $$
    \delta_t = R_{t+1} + \gamma \sum_a \pi(a \mid S_{t + 1})Q(S_{t+1}, a) - Q(S_t, A_t)
    $$
    """
    def _training(self, mdp, rng, event_listener):
        initial_avals = lambda s: {a: self.initial_q(s, a) for a in mdp.actions(s)}
        q = defaultdict2(initial_avals, initialize_defaults=True)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_terminal(s):
                # select action
                a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                # update
                na_dist = epsilon_softmax_dist(q[ns], self.rand_choose, self.softmax_temp)
                td_error = r + mdp.discount_rate*sum([q[ns][na]*p for na, p in na_dist.items()]) - q[s][a]
                q[s][a] += self.step_size*td_error
                # end of timestep
                event_listener.end_of_timestep(locals())
                s = ns
            event_listener.end_of_episode(locals())
        return q
