#!/usr/bin/env python

from easymdp3.core.util import sample_prob_dict, \
    calc_egreedy_dist

class Agent(object):
    def __init__(self, mdp):
        self.mdp = mdp

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
            traj.append((s, a, ns, r))
            s = ns
            if self.mdp.is_terminal(s):
                break
            i += 1
        return traj

class Planner(Agent):
    def solve(self):
        raise NotImplementedError

class Learner(Agent):
    def process(self, s, a, ns, r):
        raise NotImplementedError

    def episode_reset(self):
        raise NotImplementedError

    def train(self,
              episodes=20, max_steps=100,
              init_state=None, run_id=None,
              softmax_temp=None, randchoose=None,
              return_run_data=False):
        if init_state is None:
            init_state = self.mdp.get_init_state()

        run_data = []

        for e in range(episodes):
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
            self.episode_reset()

        if return_run_data:
            return run_data
        return