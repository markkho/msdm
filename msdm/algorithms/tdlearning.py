"""Temporal difference learning algorithms for discrete MDPs

"""
from functools import lru_cache
from msdm.core.algorithmclasses import Learns, Result
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.mdp.mdp import MarkovDecisionProcess
from msdm.core.mdp.policy import FunctionalPolicy
from msdm.core.distributions import DictDistribution, SoftmaxDistribution
from msdm.core.utils.dictutils import defaultdict2
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

def argmax(d, rng):
    maxv = max(d.values())
    aa = [a for a, v in d.items() if v == maxv]
    rng.shuffle(aa)
    return aa

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
        episodes : int = 100,
        step_size : float = .1,
        rand_choose : float = 0.05,
        softmax_temp : float = 0.0,
        initial_q : float = 0.0,
        seed : int = None,
        event_listener_class : TDLearningEventListener = EpisodeRewardEventListener
    ):
        """
        Generic temporal difference learning interface based on Sutton & Barto, Ch 6.

        Parameters
        ----------
        episodes : int
            The number of episodes to train
        step_size : float
            The step size of the update rule
        rand_choose : float
            Epsilon in epsilon-greedy during learning
        softmax_temp : float
            Temperature of softmax during learning (0.0 -> hardmax)
        initial_q : float or Callable[[State, Action], float]
            Initial q value or a function that returns values for a state and action. Equivalent to a heuristic.
        seed : int
            Random seed
        event_listener_class : LearningEventListener
            Event listener class
        """
        self.episodes = episodes
        self.step_size = step_size
        self.rand_choose = rand_choose
        self.softmax_temp = softmax_temp
        self.seed = seed
        if isinstance(initial_q, (float, int)):
            self.initial_q = lambda s, a: initial_q
        elif callable(initial_q):
            self.initial_q = initial_q
        else:
            raise ValueError("`inital_q` needs to be a float, int, or real-valued state-action function")
        self.event_listener_class = event_listener_class

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
        @FunctionalPolicy
        @lru_cache(maxsize=None)
        def policy(s):
            try:
                action_vals = q[s]
                maxq = max(action_vals.values())
                max_actions = [a for a in action_vals.keys() if action_vals[a] == maxq]
            except KeyError:
                max_actions = mdp.actions(s)
            return DictDistribution.uniform(max_actions)
        return policy

    def _initial_q_table(self, mdp: MarkovDecisionProcess):
        def initial_q(s, a):
            if mdp.is_absorbing(s):
                return 0.0
            return self.initial_q(s, a)
        initial_avals = lambda s: {a: initial_q(s, a) for a in mdp.actions(s)}
        q = defaultdict2(initial_avals, initialize_defaults=True)
        return q

    def train_on(self, mdp: TabularMarkovDecisionProcess):
        rng = self._init_random_number_generator()
        event_listener = self.event_listener_class()
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
    def _training(self, mdp: MarkovDecisionProcess, rng, event_listener):
        q = self._initial_q_table(mdp)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_absorbing(s):
                # select action
                a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                # update
                q[s][a] += self.step_size*(r + mdp.discount_rate*max(q[ns].values()) - q[s][a])
                # end of timestep
                event_listener.end_of_timestep(locals())
                s = ns
            event_listener.end_of_episode(locals())
        return q

class DoubleQLearning(TemporalDifferenceLearning):
    r"""
    Double Q-learning is an off-policy temporal difference control method
    that uses two q-value estimates to avoid positive bias.
    The temporal difference error in q-learning is:
    $$
    \delta_t = R_{t+1} + \gamma Q_i(S_{t+1}, \argmax_a Q_j(S_{t+1}, a)) - Q_j(S_t, A_t)
    $$
    where Q_i and Q_j are two different Q functions selected at random each update.
    """
    def _training(self, mdp: MarkovDecisionProcess, rng, event_listener):
        q1 = self._initial_q_table(mdp)
        q2 = self._initial_q_table(mdp)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_absorbing(s):
                # select action
                avals = {a: q1[s][a]*.5 + q2[s][a]*.5 for a in mdp.actions(s)}
                a = epsilon_softmax_sample(avals, self.rand_choose, self.softmax_temp, rng)
                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                # update
                if rng.random() > .5:
                    td_error = r + mdp.discount_rate*q2[ns][argmax(q1[ns], rng).pop()] - q1[s][a]
                    q1[s][a] += self.step_size*td_error
                else:
                    td_error = r + mdp.discount_rate*q1[ns][argmax(q2[ns], rng).pop()] - q2[s][a]
                    q2[s][a] += self.step_size*td_error

                # end of timestep
                event_listener.end_of_timestep(locals())
                s = ns
            event_listener.end_of_episode(locals())

        #return mean of the two estimates
        q = {}
        for s in set(q1.keys()) | set(q2.keys()):
            q[s] = {}
            for a in mdp.actions(s):
                q[s][a] = q1[s][a]*.5 +q2[s][a]*.5
        return q

class SARSA(TemporalDifferenceLearning):
    r"""
    SARSA is an on-policy temporal difference control method.
    The temporal difference error in SARSA is:
    $$
    \delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    $$
    """
    def _training(self, mdp: MarkovDecisionProcess, rng, event_listener):
        q = self._initial_q_table(mdp)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            if s not in q:
                q[s] = {a: self.initial_q(s, a) for a in mdp.actions(s)}
            a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
            while not mdp.is_absorbing(s):
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
    def _training(self, mdp: MarkovDecisionProcess, rng, event_listener):
        q = self._initial_q_table(mdp)
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_absorbing(s):
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
