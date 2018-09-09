#!/usr/bin/env python
from collections import namedtuple

from pyrlap.core.mdp import MDP
from pyrlap.core.reward_function import RewardFunction
from pyrlap.algorithms.valueiteration import ValueIteration

class FeatureGridWorld(MDP):
    def __init__(self,
                 featurelayers,
                 transition_features=None,
                 reward_features=None,
                 absorbing_features=None,
                 initstate_features=None,

                 wait_action=False,
                 wall_action=True
                 ):
        loc_features = {}
        width, height = len(featurelayers[0][0]), len(featurelayers[0])
        for layer in featurelayers:
            for y, row in layer:
                y = height - y - 1
                for x, f in row:
                    loc_features[(x, y)] = loc_features.get((x, y), set([]))
                    loc_features[(x, y)].add(f)
        self.loc_features = loc_features
        self.transition_features = transition_features
        self.reward_features = reward_features
        self.absorbing_features = absorbing_features

