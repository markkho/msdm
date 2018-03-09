#!/usr/bin/env python
from util import sample_prob_dict

import numpy as np

class Policy(object):
    def __init__(self,
                 state_action_prob=None,
                 state_action_dict=None):
        self.state_action_prob = state_action_prob
        self.state_action_dict = state_action_dict

    def get_action(self, state):
        if self.state_action_prob is not None:
            return sample_prob_dict(self.state_action_prob[state])
        elif self.state_action_dict is not None:
            return self.state_action_dict[state]

class UniformRandomPolicy(Policy):
    def __init__(self, mdp=None, markovgame=None, playername=None):
        self.mdp = mdp
        self.markovgame = markovgame
        self.playername = playername

    def get_action(self, state):
        if self.mdp is not None:
            actions = self.mdp.get_available_actions(state)
        elif self.markovgame is not None:
            actions = self.markovgame.get_available_actions(state)
            actions = actions[self.playername]
        return actions[np.random.choice(len(actions))]
