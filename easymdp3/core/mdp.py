#!/usr/bin/env python

import numpy as np

from .util import sample_prob_dict

class MDP(object):
    #=============================================#
    def get_init_state(self):
        raise NotImplementedError

    def is_terminal(self, s):
        raise NotImplementedError

    def is_terminal_action(self, a):
        raise NotImplementedError

    def transition_reward_dist(self, s, a):
        raise NotImplementedError

    def transition_dist(self, s, a):
        trdist = self.transition_reward_dist(s, a)
        tdist = {}
        for (ns, r), p in trdist.items():
            tdist[ns] = tdist.get(ns, 0) + p
        return tdist

    def transition(self, s, a):
        return sample_prob_dict(self.transition_reward_dist(s, a))[0]

    def reward(self, s=None, a=None, ns=None):
        return sample_prob_dict(self.transition_reward_dist(s, a))[1]

    def available_actions(self, s):
        raise NotImplementedError

    # =============================================#

    def get_state_features(self, s):
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError

    def get_reachable_transition_reward_functions(self,
                                                  max_states=np.inf,
                                                  init_state=None,
                                                  get_rf=True,
                                                  get_tf=True):
        if init_state is None:
            init_state = self.get_init_state()
        frontier = {init_state}
        visited = set([])
        tf = {}
        rf = {}
        while len(frontier) > 0 and len(tf) < max_states:
            s = frontier.pop()
            visited.add(s)
            tf[s] = tf.get(s, {})
            rf[s] = rf.get(s, {})
            for a in self.available_actions(s):
                tf[s][a] = tf[s].get(a, {})
                rf[s][a] = rf[s].get(a, {})
                tr_dist = self.transition_reward_dist(s, a)
                for (ns, r), p in tr_dist.items():
                    tf[s][a][ns] = p
                    rf[s][a][ns] = r
                    if ns not in visited:
                        frontier.add(ns)
        return (tf, rf)

    def transition_reward_dist(self, s, a):
        raise NotImplementedError

    def get_state_actions(self):
        raise NotImplementedError

    def get_state_action_nextstates(self):
        raise NotImplementedError

    def get_reward_dict(self):
        raise NotImplementedError

    # =============================================#
    def get_optimal_policy(self):
        raise NotImplementedError

    def get_softmax_policy(self, temp):
        raise NotImplementedError

    # ============================================= #

    def plot(self):
        raise NotImplementedError

    # =============================================#

    def solve(self):
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

    def gen_transition_dict(self, start_state=None):
        raise NotImplementedError