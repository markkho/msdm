from collections import defaultdict
from copy import deepcopy

import numpy as np

from .mdp import MDP
from .util import bidict, sample_prob_dict
from pyrlap.algorithms.valueiteration import ValueIteration

class AbstractMDP(MDP):
    def __init__(self,
                 ground_mdp: MDP,
                 abstract_state_map: bidict,
                 state_weight: dict = None,
                 action_map: bidict = None
                 ):
        """

        :param ground_mdp:

        :param abstract_state_map:
            bidict from state to abstract state

        :param state_weight:

        :param action_map:
        """

        self.ground_mdp = ground_mdp
        abstract_state_map = deepcopy(abstract_state_map)
        self.abstract_state_map = abstract_state_map
        for ts in self.ground_mdp.get_terminal_states():
            self.abstract_state_map[ts] = ts

        if state_weight is None:
            state_weight = {}

        for abs_s, g_states in abstract_state_map.inverse.items():
            if g_states[0] in state_weight:
                continue
            for s in g_states:
                state_weight[s] = 1 / len(g_states)
        self.state_weight = state_weight

        if action_map is None:
            action_map = bidict({a: a for a in ground_mdp.available_actions()})
        self.action_map = action_map

    def get_init_state(self):
        return self.abstract_state_map[self.ground_mdp.get_init_state()]

    def get_init_states(self):
        init_states = set([])
        for s in self.ground_mdp.get_init_states():
            init_states.add(self.abstract_state_map[s])
        return list(init_states)

    def is_terminal(self, abs_s):
        s = self.abstract_state_map.inverse[abs_s][0]
        return self.ground_mdp.is_terminal(s)

    def is_terminal_action(self, abs_a):
        return self.ground_mdp.is_terminal_action(self.action_map[abs_a][0])

    def available_actions(self, s=None):
        return self.ground_mdp.available_actions()

    def transition_dist(self, s, a):
        abs_s, abs_a = s, a
        ground_s_set = self.abstract_state_map.inverse[abs_s]
        ground_a = self.action_map.inverse[abs_a]
        tdist = defaultdict(int)
        for s in ground_s_set:
            a = ground_a[0]
            sweight = self.state_weight[s]
            ground_tdist = self.ground_mdp.transition_dist(s, a)
            for ns, p in ground_tdist.items():
                abs_ns = self.abstract_state_map[ns]
                tdist[abs_ns] += sweight*p
        return tdist

    def transition(self, s, a):
        return sample_prob_dict(self.transition_dist(s, a))

    def reward(self, s=None, a=None, ns=None):
        abs_s, abs_a, abs_ns = s, a, ns
        abs_r = 0

        ground_ns = self.abstract_state_map.inverse[abs_ns]
        for s in self.abstract_state_map.inverse[abs_s]:
            a = abs_a[0] #assume no action abstraction for now
            sweight = self.state_weight[s]
            cond_nsdist = {}
            normalizer = 0
            for ns, p in self.ground_mdp.transition_dist(s, a).items():
                if ns not in ground_ns:
                    continue
                cond_nsdist[ns] = p
                normalizer += p
            cond_nsdist = {ns: p/normalizer for ns, p in cond_nsdist.items()}

            for ns, p in cond_nsdist.items():
                abs_r += sweight*p*self.ground_mdp.reward(s=s, a=a, ns=ns)
        return abs_r

    def reward_dist(self, s=None, a=None, ns=None):
        samples = 1

        abs_s, abs_a, abs_ns = s, a, ns
        abs_rdist = defaultdict(int)

        for _ in range(samples):
            r = self.reward(s=abs_s, a=abs_a, ns=abs_ns)
            abs_rdist[r] += 1
        abs_rdist = {r: c/samples for r, c in abs_rdist.items()}
        return abs_rdist

    def transition_reward_dist(self, s, a):
        return self._cond_tfrf_to_joint(s, a)

    def solve(self,
              discount_rate,
              softmax_temp=0.0,
              randchoose=0.0,
              **kwargs):
        planner = ValueIteration(mdp=self,
                                 discount_rate=discount_rate,
                                 softmax_temp=softmax_temp,
                                 randchoose=randchoose,
                                 **kwargs)
        planner.solve()
        return planner

    #====================================#
    def calc_stationary_dist(self, policy):
        """
        This takes a deterministic policy over the abstract MDP and composes
        it with the ground MDP. It then calculates the stationary distribution
        (occupancy distribution) under that policy.
        """
        return self._gen_stationary_dist_by_sampling(policy)

    def _gen_stationary_dist_by_sampling(self, policy,
                                         trajectories=500,
                                         max_trajlen=20):
        occ = {s: 0 for s in self.ground_mdp.get_states()}

        for _ in range(trajectories):
            s = self.ground_mdp.get_init_state()
            for _ in range(max_trajlen):
                occ[s] += 1
                if self.ground_mdp.is_terminal(s):
                    break
                abs_s = self.abstract_state_map[s]
                a = policy.act(abs_s)
                ns = self.ground_mdp.transition(s, a)
                s = ns
        occ_probs = {}
        for abs_s, ground_ss in self.abstract_state_map.inverse.items():
            counts = np.array([occ[s] for s in ground_ss])
            if np.sum(counts) == 0:
                continue
            probs = counts / np.sum(counts)
            occ_probs.update(dict(zip(ground_ss, probs)))
        return occ_probs

    def _update_state_weight(self, state_weight):
        self.state_weight = state_weight

