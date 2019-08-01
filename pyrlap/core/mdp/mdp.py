#!/usr/bin/env python

import numpy as np
import copy

from pyrlap.core.util import sample_prob_dict, SANSRTuple, SANSTuple
from pyrlap.core.transition_function import TransitionFunction

class MDP(object):
    #=============================================#
    def get_init_state(self):
        raise NotImplementedError

    def get_init_state_dist(self):
        raise NotImplementedError

    def get_init_states(self):
        return []

    def is_terminal(self, s):
        raise NotImplementedError

    def is_absorbing(self, s):
        raise NotImplementedError

    def is_any_terminal(self, s):
        raise NotImplementedError

    def is_terminal_action(self, a):
        raise NotImplementedError

    def get_terminal_states(self):
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
        raise NotImplementedError

    def reward_dist(self, s=None, a=None, ns=None):
        raise NotImplementedError

    def available_actions(self, s=None):
        raise NotImplementedError

    # =============================================#
    """
    There are multiple ways to represent the probability distribution
    P(ns, r | s, a). These are functions designed to let you define
    it in one way and then translate it into the other ways for use in
    different algorithms.
    """

    def _cond_tfrf_to_joint(self, s=None, a=None):
        """
        Requires that transition_dist(s, a) and reward_dist(s, a, ns)
        are defined.

        :param s:
        :param a:
        :return:
        """
        tdist = self.transition_dist(s, a)
        trdist = {}
        for ns, tp in tdist.items():
            rdist = self.reward_dist(s, a, ns)
            for r, rp in rdist.items():
                trdist[(ns, r)] = tp * rp
        return trdist

    # ============================================= #

    def get_state_features(self, s):
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError

    def get_non_terminal_states(self):
        states = []
        for s in self.get_states():
            if not (
                self.is_absorbing(s) or
                self.is_terminal(s)
            ):
                states.append(s)
        return states

    def get_reachable_transition_reward_functions(self,
                                                  max_states=np.inf,
                                                  init_state=None,
                                                  init_states=None):
        if init_states is None:
            init_states = []

        if init_state is None:
            init_state = self.get_init_state()

        init_states.append(init_state)
        frontier = set(init_states)
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
                    tf[s][a][ns] = tf[s][a].get(ns, 0)
                    tf[s][a][ns] += p

                    rf[s][a][ns] = rf[s][a].get(ns, 0)
                    rf[s][a][ns] += r*p
                    if ns not in visited:
                        frontier.add(ns)
        return (TransitionFunction(tf), rf)

    def get_reachable_states(self, max_states=np.inf,
                             init_state=None, init_states=None):
        return self.get_reachable_transition_reward_functions(
            max_states, init_state, init_states
        )[0].keys()

    def get_state_actions(self):
        raise NotImplementedError

    def get_state_action_nextstates(self):
        raise NotImplementedError
    
    def iterate_sans_prob(self):
        for s in self.get_states():
            for a in self.available_actions(s):
                for ns, prob in self.transition_dist(s, a).items():
                    yield (s, a, ns, prob)

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
        tf, rf = self.get_reachable_transition_reward_functions(
            init_state=start_state
        )
        return tf

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

    def as_matrices(self):
        aa = self.available_actions()
        aa_i = {a: i for i, a in enumerate(aa)}
        ss = self.get_states()
        ss_i = {s: i for i, s in enumerate(ss)}
        tf = np.zeros((len(ss), len(aa), len(ss)), dtype=np.float32)
        rf = np.zeros((len(ss), len(aa), len(ss)), dtype=np.float32)
        for s in ss:
            for a in aa:
                ns_dist = self.transition_dist(s, a)
                for ns, p in ns_dist.items():
                    tf[ss_i[s], aa_i[a], ss_i[ns]] = p
                    rf[ss_i[s], aa_i[a], ss_i[ns]] = self.reward(s, a, ns)
        s0 = self.get_init_state_dist()
        s0 = np.array([s0.get(s, 0) for s in ss], dtype=np.float32)
        non_term = set(self.get_non_terminal_states())
        nt_states = np.array([1 if s in non_term else 0 for s in ss])
        return {
            'tf': tf, 'rf': rf, 's0': s0, 'ss': ss, 'aa': aa,
            'nt_states': nt_states
        }

    def is_valid_transition(self, s, a, ns, *args, **kwargs):
        ns_dist = self.transition_dist(s, a)
        return ns in ns_dist

    def is_valid_trajectory(self, traj):
        for step in traj:
            if not self.is_valid_transition(*step):
                return False
        return True