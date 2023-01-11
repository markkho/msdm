"""RMAX learning algorithm for MDPs"""
from functools import lru_cache
from pyclbr import Function
import random
from types import SimpleNamespace

import numpy as np

from msdm.core.distributions import DictDistribution
from msdm.core.algorithmclasses import Learns, Result
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.mdp.policy import FunctionalPolicy
from msdm.core.utils.funcutils import cached_property
from abc import abstractmethod, ABC

class RMAXEventListener(ABC):
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

class EpisodeRewardEventListener(RMAXEventListener):
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

class RMAX(Learns):
    def __init__(
        self,
        episodes : int = 100,
        rmax : float = 1.0,
        num_transition_samples : int = 3,
        bellman_convergence_diff : float = 1e-5,
        seed : int = None,
        event_listener_class : RMAXEventListener = EpisodeRewardEventListener,
    ):
        """
        RMAX learning algorithm based on the pseudocode in [Strehl, Li and Littman 2009]
        https://jmlr.org/papers/volume10/strehl09a/strehl09a.pdf
        Although the algorithm is first proposed in [Brafman and Tennenholtz 2002]
        https://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf

        Parameters
        ----------
        episodes : int
            The number of episodes to train
        rmax : float
            The maximum reward that can be obtained in the MDP
        num_transition_samples : int
            The number of samples of each empirical transition to use in building an empirical model of the MDP
        bellman_convergence_diff : float
            The convergence threshold for the Bellman equation
        seed : int
            Random seed
        event_listener_class : LearningEventListener
            Event listener class
        """
        self.episodes = episodes
        self.rmax = rmax
        self.m = num_transition_samples
        self.bellman_convergence_diff = bellman_convergence_diff
        self.seed = seed
        self.event_listener_class = event_listener_class

    def _init_random_number_generator(self):
        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random
        return rng

    def _create_policy(self, mdp : TabularMarkovDecisionProcess, q):
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

    def _create_q(self, q_matrix, mdp : TabularMarkovDecisionProcess):
        """create a dictionary q from a q matrix"""
        index_to_state = dict(enumerate(mdp.state_list))
        index_to_action = dict(enumerate(mdp.action_list))
        q = {}

        for si in range(q_matrix.shape[0]):
            s = index_to_state[si]
            q[s] = {}
            for ai in range(q_matrix.shape[1]):
                a = index_to_action[ai]
                q[s][a] = q_matrix[si, ai]
        return q
    
    def _init_training(self, mdp : TabularMarkovDecisionProcess):
        """initialize training process by creating the data structure to build an empirical model of the MDP"""
        assert self.rmax == np.max(mdp.reward_matrix)

        self.n_states = len(mdp.reachable_states())
        self.n_actions = len(mdp.action_list)
        self.rewards = np.zeros((self.n_states, self.n_actions))  # used to record the rewards R(s, a) seen
        self.transitions = np.zeros((self.n_states, self.n_actions, self.n_states))  # used to count the number of (s, a, s') transitions seen
        self.s_a_counts = np.zeros((self.n_states, self.n_actions))  # used to count the number of (s, a) transitions seen
        
        self.q_matrix = np.ones((self.n_states, self.n_actions)) * self.rmax * 1/(1-mdp.discount_rate)

    def _act(self, state, rng):
        """advance one step during training by picking an action"""
        # Grad random action in case all Q values equal
        if np.all(self.q_matrix[state] == self.q_matrix[state][0]):
            a = rng.choice(range(self.n_actions))
        else:
            a = np.argmax(self.q_matrix[state])
        return a
    
    def _observe(self, state, action, reward, next_state, gamma):
        """observe a transition and update the empirical model of the MDP if necessary"""
        if self.s_a_counts[state, action] < self.m:
            self.rewards[state, action] += reward
            self.s_a_counts[state, action] += 1
            self.transitions[state, action, next_state] += 1

            if self.s_a_counts[state, action] == self.m:
                self._value_iteration(gamma)
                
    def _value_iteration(self, gamma):
        """do value iteration to solve the Bellman equation during one update step"""
        # mask for update
        mask = self.s_a_counts >= self.m
        pseudo_count = np.where(self.s_a_counts == 0, 1, self.s_a_counts)  # avoid divide by zero

        # build the reward model
        empirical_reward_mat = self.rewards / pseudo_count

        # build the transition model: assume self-loop if there's not enough data
        # assume a self-loop if there's not enough data
        empirical_transition_mat = self.transitions / pseudo_count[:, :, None]
        # only masked positions should be trusted, otherwise self transition
        empirical_transition_mat[~mask] = self._self_transition_mat[~mask]
        assert np.all(np.isclose(empirical_transition_mat.sum(axis=-1), 1)), empirical_transition_mat.sum(axis=-1)

        # compute the update for every (s, a), but only apply the ones that needed with a mask
        while True:
            v = np.max(self.q_matrix, axis=-1)
            new_q = empirical_reward_mat + gamma * np.einsum("san,n->sa", empirical_transition_mat, v)
            if np.all(np.abs(self.q_matrix[mask] - new_q[mask]) < self.bellman_convergence_diff):
                break
            self.q_matrix[mask] = new_q[mask]

    @cached_property
    def _self_transition_mat(self):
        self_transition_mat = np.zeros_like(self.transitions)
        self_transition_mat[np.arange(self.n_states), :, np.arange(self.n_states)] = 1
        return self_transition_mat

    def _training(
        self,
        mdp : TabularMarkovDecisionProcess,
        rng : random.Random,
        event_listener : RMAXEventListener
    ):
        """This is the main training loop. It should return
        a nested dictionary. Specifically, a dictionary with
        states as keys and action-value dictionaries as values."""
        index_to_action = dict(enumerate(mdp.action_list))
        for ep in range(self.episodes):
            s = mdp.initial_state_dist().sample(rng=rng)
            while not mdp.is_absorbing(s):
                # select action
                ai = self._act(mdp.state_list.index(s), rng)
                a = index_to_action[ai]
                # transition to next state
                ns = mdp.next_state_dist(s, a).sample(rng=rng)
                r = mdp.reward(s, a, ns)
                # update
                self._observe(mdp.state_list.index(s), ai, r, mdp.state_list.index(ns), gamma=mdp.discount_rate)
                # end of time step
                event_listener.end_of_timestep(locals())
                s = ns
                # end of episode
            event_listener.end_of_episode(locals())
        
        return self.q_matrix

    def train_on(self, mdp: TabularMarkovDecisionProcess):
        rng = self._init_random_number_generator()
        self._init_training(mdp)
        event_listener = self.event_listener_class()
        q_matrix = self._training(mdp, rng, event_listener)
        q = self._create_q(q_matrix, mdp)
        return Result(
            q_values=q,
            policy=self._create_policy(mdp, q),
            event_listener_results=event_listener.results(),
        )
