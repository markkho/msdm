#!/usr/bin/env python

import logging, warnings
from random import randint
import sys
from collections import Mapping, defaultdict

import numpy as np

from pyrlap.core.util import sample_prob_dict, calc_esoftmax_dist, SANSRTuple
from pyrlap.core.mdp.mdp import MDP as MDPClass

logger = logging.getLogger(__name__)

class ValueFunction(Mapping):
    def __init__(self, vf_dict):
        self.vf_dict = vf_dict

    def __getitem__(self, key):
        return self.vf_dict[self.__keytransform__(key)]

    def __delitem__(self, key):
        del self.vf_dict[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.vf_dict)

    def __len__(self):
        return len(self.vf_dict)

    def __keytransform__(self, key):
        return key

class Agent(object):
    def __init__(self, mdp: MDPClass, policy_dict: dict = None):
        self.mdp = mdp
        self.policy_dict = policy_dict

    def __getitem__(self, s):
        return self.act(s)

    def act_dist(self, s, softmax_temp=None, randchoose=None):
        if self.policy_dict is None:
            raise NotImplementedError
        return self.policy_dict[s]

    def to_dict(self, **kwargs) -> dict:
        policy = {}
        for s in self.mdp.get_states():
            policy[s] = self.act_dist(s, **kwargs)
        return policy

    def as_matrix(self):
        mats = self.mdp.as_matrices()
        ss = mats['ss']
        aa = mats['aa']
        policy = np.zeros((len(ss), len(aa)))
        pol_dict = self.to_dict()
        for si, s in enumerate(ss):
            if s not in pol_dict:
                continue
            for ai, a in enumerate(aa):
                policy[si][ai] = pol_dict[s].get(a, 0)
        return policy

    def act(self, s, softmax_temp=None, randchoose=None):
        adist = self.act_dist(s, softmax_temp, randchoose)
        return sample_prob_dict(adist)

    def run(self,
            softmax_temp=None,
            randchoose=None,
            init_state=None,
            max_steps=100):
        traj = []
        if init_state is None:
            init_state = self.mdp.get_init_state()
        s = init_state
        i = 0
        while i < max_steps:
            a = self.act(s, softmax_temp=softmax_temp, randchoose=randchoose)
            if hasattr(self.mdp, 'transition_reward'):
                ns, r = self.mdp.transition_reward(s=s, a=a)
            else:
                ns = self.mdp.transition(s, a)
                r = self.mdp.reward(s, a, ns)
            traj.append(SANSRTuple(s, a, ns, r))
            s = ns
            if self.mdp.is_terminal(s):
                break
            i += 1
        return traj

    def value(self, discount_rate=.99) -> ValueFunction:
        """
        return the value function of this policy - basically just do
        policy evaluation
        """
        sr = self.successor_representation(
            discount_rate=discount_rate,
            discounted=True,
            normalize=False,
            return_matrix=True
        )
        mdp_mat = self.mdp.as_matrices()
        rf = mdp_mat['rf']
        tf = mdp_mat['tf']
        ss = mdp_mat['ss']
        pol = self.as_matrix()
        s_rf = np.einsum("san,san,sa->s",tf,rf,pol)
        v = np.einsum("sn,n->s",sr,s_rf)
        return ValueFunction({s: val for s, val in zip(ss, v)})

    def calc_occupancy(self,
                       discount_rate=.99,
                       discounted=False,
                       normalize=False):
        sr = self.successor_representation(
            discount_rate=discount_rate,
            discounted=discounted,
            normalize=normalize,
            return_matrix=True
        )
        s0 = self.mdp.as_matrices()['s0']
        ss = self.mdp.as_matrices()['ss']
        occ = np.einsum("s,sn->n", s0, sr)
        occ = {s: o for s, o in zip(ss, occ)}
        return occ

    def successor_representation(self,
                                 discount_rate=.99,
                                 discounted=False,
                                 normalize=False,
                                 return_matrix=False):
        mdp_mat = self.mdp.as_matrices()
        tf = mdp_mat['tf']
        ss = mdp_mat['ss']

        mp = np.einsum("san,sa->sn", tf, self.as_matrix())

        # Calculate discounted or undiscounted successor representation
        if not discounted:
            if np.linalg.cond(np.eye(len(ss)) - mp) < 1/sys.float_info.epsilon:
                sr = np.linalg.inv(np.eye(len(ss)) - mp)
            else:
                warnings.warn(
                    "Undiscounted transition matrix is singular. "+
                    ("Calculating discounted occupancy dr = %.2f" % discount_rate)
                )
                sr = np.linalg.inv(np.eye(len(ss)) - discount_rate*mp)
        if discounted:
            sr = np.linalg.inv(np.eye(len(ss)) - discount_rate * mp)

        if normalize:
            sr_norm = np.einsum("sn->s", sr)
            sr = np.einsum("s,sn->sn", 1/sr_norm, sr)

        if return_matrix:
            return sr

        sr = {s: dict(zip(ss, s_sr)) for s, s_sr in zip(ss, sr)}
        return sr

class RandomAgent(Agent):
    def act_dist(self, s, softmax_temp=None, randchoose=None):
        aa = self.mdp.available_actions(s)
        return dict(zip(aa, [1/len(aa) for _ in range(len(aa))]))

class Planner(Agent):
    def solve(self):
        raise NotImplementedError

class ActionValueFunctionAgent(Agent):
    def __init__(self,
                 mdp,
                 action_value_function,
                 softmax_temp=0.0,
                 randchoose=0.0):
        Agent.__init__(self, mdp)
        self.action_value_function = action_value_function
        self.softmax_temp = softmax_temp
        self.randchoose = randchoose

    def act_dist(self, s, softmax_temp=None, randchoose=None):
        if softmax_temp is None:
            softmax_temp = self.softmax_temp
        if randchoose is None:
            randchoose = self.randchoose

        return calc_esoftmax_dist(self.action_value_function[s],
                                  temp=softmax_temp,
                                  randchoose=randchoose)

class Learner(Agent):
    def process(self, s, a, ns, r):
        raise NotImplementedError

    def episode_reset(self):
        raise NotImplementedError

    def train(self,
              episodes=20, max_steps=100,
              init_state=None, run_id=None,
              softmax_temp=None, randchoose=None,
              run_data=None,
              return_run_data=False):
        if init_state is None:
            init_state = self.mdp.get_init_state()

        start_ep = 0
        if run_data is None:
            run_data = []

        if len(run_data) > 0:
            start_ep = run_data[-1]['episode'] + 1

        for e in range(start_ep, episodes + start_ep):
            s = init_state
            for t in range(max_steps):
                a = self.act(s,
                             softmax_temp=softmax_temp,
                             randchoose=randchoose)
                if hasattr(self.mdp, 'transition_reward'):
                    ns, r = self.mdp.transition_reward(s=s, a=a)
                else:
                    ns = self.mdp.transition(s=s, a=a)
                    r = self.mdp.reward(s=s, a=a, ns=ns)

                if return_run_data:
                    run_data.append({
                        'episode': e, 'timestep': t,
                        's': s, 'a': a, 'ns': ns, 'r': r
                    })
                    if run_id is not None:
                        run_data[-1]['run_id'] = run_id
                self.process(s, a, ns, r)
                s = ns
                if self.mdp.is_terminal(ns):
                    break
            if e % 100 == 0:
                logger.debug('run: %d ; steps: %d' % (e, t))

            self.episode_reset()

        if return_run_data:
            return run_data
        return