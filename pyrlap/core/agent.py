#!/usr/bin/env python

import logging

from pyrlap.core.util import sample_prob_dict, calc_esoftmax_dist, SANSRTuple

logger = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, mdp):
        self.mdp = mdp

    def __getitem__(self, s):
        return self.act(s)

    def act_dist(self, s, softmax_temp=None, randchoose=None):
        raise NotImplementedError

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