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

    def value(self,
              discount_rate=.99,
              max_iterations=250,
              converge_delta=.01) -> ValueFunction:
        """
        return the value function of this policy - basically just do
        policy evaluation
        """
        vf = {}

        for i in range(max_iterations):
            change = 0
            vf_temp = {}
            for s in self.mdp.get_states():
                s_val = 0
                adist = self.act_dist(s)
                for a, aprob in adist.items():
                    ns_dist = self.mdp.transition_dist(s, a)
                    for ns, nsprob in ns_dist.items():
                        r = self.mdp.reward(s, a, ns)
                        v = vf.get(ns, 0)
                        s_val += (r + discount_rate*v)*aprob*nsprob
                vf_temp[s] = s_val
                change = max(change, abs(vf_temp[s] - vf.get(s, 0)))
            vf = vf_temp
            logger.debug('iteration: %d   change: %.2f' % (i, change))
            if change < converge_delta:
                break

        if change >= converge_delta:
            warnings.warn(
                "Policy evaluation did not converge after %d iterations (delta=%.2f)" \
                % (i, change))
        return ValueFunction(vf)

    def successor_representation(self,
                                 state=None,
                                 init_state_dist=None,
                                 discount_rate=.99,
                                 normalize=False):
        mdp = self.mdp
        states = sorted(mdp.get_reachable_states())
        if init_state_dist is None:
            if state is None:
                init_state_dist = mdp.get_init_state_dist()
            else:
                init_state_dist = {state: 1}

        # calculate reward process resulting from policy
        rp_matrix = np.zeros((len(states), len(states)))
        for s in states:
            adist = self.act_dist(s)
            if mdp.is_terminal(s):
                continue
            ns_dist = defaultdict(float)
            for a, p in adist.items():
                nsd = mdp.transition_dist(s, a)
                for ns, nsp in nsd.items():
                    ns_dist[ns] += nsp * p
            assert ((sum(ns_dist.values()) - 1) < .00001)  # sum to 1
            for ns, nsp in ns_dist.items():
                rp_matrix[states.index(s), states.index(ns)] = nsp

        m_tot = np.eye(len(states)) - rp_matrix
        if np.linalg.cond(m_tot) < 1/sys.float_info.epsilon:
            m_tot = np.linalg.inv(m_tot)
        else:
            warnings.warn(
                "Undiscounted transition matrix is singular. "+
                ("Calculating discounted occupancy dr = %.2f" % discount_rate)
            )
            m_tot = np.eye(len(states)) - discount_rate*rp_matrix
            m_tot = np.linalg.inv(m_tot)

        init_states, init_probs = list(zip(*init_state_dist.items()))
        init_idx = [states.index(s) for s in init_states]
        occupancy = m_tot[init_idx, :].T @ init_probs

        terminal_states = \
            [states.index(ts) for ts in self.mdp.get_terminal_states()
             if ts in states]
        occupancy[terminal_states] = 0
        if normalize:
            occupancy = occupancy/np.sum(occupancy)
        occupancy = dict(zip(states, occupancy))
        return occupancy

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