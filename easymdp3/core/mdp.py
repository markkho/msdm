#!/usr/bin/env python
from __future__ import division
import random

import numpy as np

from ..value_iteration import VI
from util import sample_prob_dict


class MDP(object):
    def __init__(self):
        pass

    def is_terminal(self, s):
        raise NotImplementedError

    def transition_dist(self, s, a):
        raise NotImplementedError

    def transition(self, s, a):
        raise NotImplementedError

    def reward(self, s=None, a=None, ns=None):
        raise NotImplementedError

    def available_actions(self, s):
        raise NotImplementedError

    def state_hasher(self, state):
        raise NotImplementedError

    def state_unhasher(self, hashed):
        raise NotImplementedError

    def get_optimal_policy(self):
        raise NotImplementedError

    def get_softmax_function(self, temp):
        raise NotImplementedError

    def get_init_state(self):
        raise NotImplementedError

    def gen_state_actions(self):
        raise NotImplementedError

    def gen_state_action_nextstates(self):
        raise NotImplementedError

    def get_state_features(self, s):
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError

    def gen_reward_dict(self):
        raise NotImplementedError

    def calc_trajectory_return(self, traj, init_state=None, discount=1):
        value = 0

        if len(traj[0]) == 1:
            ns = init_state
        
        for tup in traj:
            if len(tup) == 3:
                s, a, ns = tup
            elif len(tup) == 2:
                s, a = tup
                ns = self.transition(s, a)
            elif len(tup) == 1:
                a = tup
                s = ns
                ns = self.transition(s, a)
            value += self.reward(s=s, a=a, ns=ns)*discount
        return value

    def build_transition_graph(self, init_state, max_nodes=np.inf):
        graph = {}
        frontier = [init_state, ]

        while len(graph) < max_nodes:
            s = frontier.pop()

    def gen_transition_dict(self, start_state=None):
        tf = {}
        for s in self.get_states():
            tf[s] = {}
            for a in self.available_actions(s):
                tf[s][a] = self.transition_dist(s, a)
        return tf

    def solve(self, start_state=None, rf=None, **kwargs):
        if 'discount_rate' in kwargs:
            kwargs['gamma'] = kwargs['discount_rate']
            del kwargs['discount_rate']

        if 'gamma' not in kwargs and self.discount_rate is not None:
            kwargs['gamma'] = self.discount_rate
        if rf is None:
            rf = self.gen_reward_dict()
        else:
            state_actions = self.gen_state_actions()
            state_action_nextstates = self.gen_state_action_nextstates()
            rf = rf.gen_reward_dict(
                states=self.states,
                state_actions=state_actions,
                state_action_nextstates=state_action_nextstates
            )
        tf = self.gen_transition_dict()
        op, vf, av = VI(rf, tf, init_state=start_state, **kwargs)
        self.optimal_policy = op
        self.value_function = vf
        self.action_value_function = av

        self.solved = True

    def run(self, policy=None, init_state=None, max_steps=25, temp=1):
        traj = []
        if init_state is None:
            init_state = self.get_init_state()
        if policy is None:
            policy = self.get_softmax_function(temp)
        s = init_state
        i = 0
        while i < max_steps:
            a = sample_prob_dict(policy[s])
            ns = self.transition(s, a)
            r = self.reward(s, a, ns)
            traj.append((s, a, ns, r))
            s = ns
            if self.is_terminal(s):
                break
            i += 1
        return traj