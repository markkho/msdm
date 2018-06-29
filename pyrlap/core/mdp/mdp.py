#!/usr/bin/env python

import numpy as np
import copy

from pyrlap.core.util import sample_prob_dict, SANSRTuple, SANSTuple

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
        if ns is not None:
            rdist = {}
            norm = 0
            for (ns_, r), p in self.transition_reward_dist(s, a).items():
                if ns_ == ns:
                    rdist[r] = rdist.get(r, 0) + p
                    norm += p
            rdist = {r: p/norm for r, p in rdist.items()}
            return sample_prob_dict(rdist)
        return sample_prob_dict(self.transition_reward_dist(s, a))[1]

    def reward_dist(self, s=None, a=None, ns=None):
        raise NotImplementedError

    def available_actions(self, s):
        raise NotImplementedError

    # =============================================#

    def get_state_features(self, s):
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError

    def get_reachable_transition_reward_functions(self,
                                                  max_states=np.inf,
                                                  init_state=None):
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

    def solve(self, discount_rate):
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

    def run_policy(self, policy, init_state=None):
        if init_state is None:
            init_state = self.get_init_state()
        traj = []
        s = init_state
        while not self.is_terminal(s):
            a = policy(s)
            ns = self.transition(s, a)
            r = self.reward(s, a, ns)
            traj.append(SANSRTuple(s, a, ns, r))
            s = ns
        return traj

    def enumerate_trajs(self, init_states=None, depth=4,
                        no_deterministic_loops=True):
        if init_states is None:
            init_states = [self.get_init_state(), ]
        trajs = set([])
        frontier = \
            set([tuple([SANSTuple(None, None, s),])for s in init_states])
        while len(frontier) > 0:
            pre_traj = list(frontier.pop())
            s = pre_traj[-1].ns
            if pre_traj[-1].s is None:
                pre_traj = []
            for a in self.available_actions(s):
                for ns, tp in self.transition_dist(s, a).items():
                    if no_deterministic_loops and ns == s and tp == 1:
                        continue
                    new_pre_traj = copy.copy(pre_traj)
                    new_pre_traj.append(SANSTuple(s, a, ns))
                    if self.is_terminal(ns):
                        trajs.add(tuple(new_pre_traj))
                    elif len(new_pre_traj) == depth:
                        trajs.add(tuple(new_pre_traj))
                    else:
                        frontier.add(tuple(new_pre_traj))
        return list(trajs)