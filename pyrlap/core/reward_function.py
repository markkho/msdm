#!/usr/bin/env python
import copy
import logging
logger = logging.getLogger(__name__)

import numpy as np


class RewardFunction(object):
    """
    TODO: implement state-action and state-action-nextstate feature rfs
    
    state_features : dict with mapping from states to lists of features
    feature_rewards : dict with mapping from features to reward values
    
    Reward is simply the sum of all features (for now). This implementation
    represents reward functions based on either states; states and actions, or
    states, actions, and nextstates. Orthogonally, it can represent them
    in tabular form, or as sums of features (over either states, states/actions,
    or states/actions/nextstates.
    """
    def __init__(self,
                 state_features=None,
                 state_rewards=None, #this is a deprecated argument
                 reward_dict=None,
                 feature_rewards=None,
                 default_reward=0,
                 terminal_states=None,
                 terminal_state_reward=0,
                 step_cost=0,
                 rmax=None,
                 cache_rewards=True):
        if terminal_states is None:
            terminal_states = [(-1, -1), (-2, -2)]
        if state_rewards is not None:
            reward_dict = state_rewards

        self.terminal_states = tuple(sorted(terminal_states))
        self.terminal_state_reward = terminal_state_reward
        self.default_reward = default_reward
        
        if (state_features is not None) and (feature_rewards is not None):
            self.state_features = state_features
            self.feature_rewards = feature_rewards
            self.type = 'state_feature_based'
        elif reward_dict is not None:
            self.reward_dict = copy.deepcopy(reward_dict)
            if type(list(reward_dict.values())[0]) is dict:
                if type(list(reward_dict.values())[0].values()[0]) is dict:
                    self.type = 'state_action_nextstate_dict'
                else:
                    self.type = 'state_action_dict'
            else:
                self.type = 'state_dict'
        else:
            self.reward_dict = {}
            self.type = 'state_dict'
        
        if self.type == 'state_dict':
            for ts in terminal_states:
                self.reward_dict[ts] = terminal_state_reward

        self.step_cost = step_cost

        #set rmax
        if rmax is None:
            if self.type == 'state_dict':
                rs = list(self.reward_dict.values()) + [default_reward,]
                rmax = max(rs)
            elif self.type == 'state_action_dict':
                rmax = -np.inf
                for s, ar in self.reward_dict.items():
                    for a, r in ar.items():
                        rmax = max(rmax, r)
            elif self.type == 'state_feature_based':
                fr = np.array(list(self.feature_rewards.values()))
                pos_fr = fr[fr > 0]
                if (len(pos_fr) == 0):
                    pos_fr = [max(fr),]
                rmax = np.sum(pos_fr)
            else:
                raise ValueError("Cannot set Rmax")
        self.rmax = rmax

        self.reward_cache = {}
        self.cache_rewards = cache_rewards
        
    def reward(self, s=None, a=None, ns=None):
        if self.type == 'state_dict':
            reward = self.reward_dict.get(ns, self.default_reward)

        elif self.type == "state_action_dict":
            if s not in self.reward_dict:
                reward = self.default_reward
            else:
                reward = self.reward_dict[s].get(a, self.default_reward)

        elif self.type == "state_action_nextstate_dict":
            if s not in self.reward_dict:
                reward = self.default_reward
            elif a not in self.reward_dict[s]:
                reward = self.default_reward
            else:
                reward = self.reward_dict[s][a].get(ns, self.default_reward)

        elif self.type == 'state_feature_based':
            if ns in self.terminal_states:
                reward = self.terminal_state_reward

            elif self.cache_rewards:
                if ns not in self.reward_cache:
                    fs = self.state_features.get(ns, [])
                    r = 0
                    for f in fs:
                        r += self.feature_rewards.get(f, self.default_reward)
                    self.reward_cache[ns] = r
                reward = self.reward_cache[ns]
            else:
                fs = self.state_features.get(ns, [])
                reward = np.sum([self.feature_rewards[f] for f in fs])

        elif self.type == 'state_action_feature_based':
            pass

        elif self.type == 'state_action_nextstate_feature_based':
            pass
        
        if ns in self.terminal_states:
            return reward
        else:
            return reward + self.step_cost

    def gen_reward_dict(self, states=None, state_actions=None,
                        state_action_nextstates=None, tf=None,
                        include_actions=False, include_nextstates=False):

        # ================================================ #
        #  Generate a state-action-nextstate rf dictionary #
        # ================================================ #
        if (include_actions and include_nextstates) \
                or self.type in ['state_action_nextstate_dict',
                                 'state_action_nextstate_feature_based']:
            rf = {}
            for s, a_ns in state_action_nextstates.items():
                rf[s] = {}
                for a, nstates in a_ns.items():
                    rf[s][a] = {}
                    for ns in nstates:

                        #=========================================#
                        #      Handle the different rf types      #
                        #=========================================#
                        if self.type in ['state_dict', 'state_feature_based']:
                            rf[s][a][ns] = self.reward(ns=ns)

                        elif self.type in ['state_action_dict',
                                           'state_action_feature_based']:
                            rf[s][a][ns] = self.reward(s=s, a=a)

                        elif self.type in ['state_action_nextstate_dict',
                                           'state_action_nextstate_feature_based']:
                            rf[s][a][ns] = self.reward(s=s, a=a, ns=ns)

                        else:
                            raise ValueError("Undefined reward function dictionary!")

        # ================================================ #
        #       Generate a state-action rf dictionary      #
        # ================================================ #
        elif include_actions or self.type in ['state_action_dict',
                                              'state_action_feature_based']:
            rf = {}
            # ======================================== #
            #      Handle the different rf types       #
            # ======================================== #
            if self.type in ['state_dict', 'state_feature_based']:
                for s, a_ns in state_action_nextstates.items():
                    rf[s] = {}
                    for a, nstates in a_ns.items():
                        if len(nstates) > 1:
                            raise ValueError("Undefinable reward function dictionary!")
                        rf[s][a] = self.reward(ns=nstates[0])

            elif self.type in ['state_action_dict',
                               'state_action_feature_based']:
                for s, actions in state_actions.items():
                    rf[s] = {}
                    for a in actions:
                        rf[s][a] = self.reward(s=s, a=a)
            else:
                raise ValueError("Undefined reward function dictionary!")

        # ================================================ #
        #           Generate a state rf dictionary         #
        # ================================================ #
        elif self.type in ['state_dict', 'state_feature_based']:
            rf = {ns: self.reward(ns=ns) for ns in states}

        else:
            raise ValueError("Undefined reward function dictionary!")

        return rf


    def __hash__(self):
        try:
            return self.hash
        except AttributeError:
            pass

        #todo write a test for this hash function
        myhash = [self.type,
                  self.terminal_state_reward,
                  self.terminal_states,
                  self.default_reward,
                  self.step_cost]

        if self.type == 'state_dict':
            myhash.extend([
                tuple(sorted(self.reward_dict.items())),
            ])
        else:
            myhash.extend([
                False,
            ])

        if self.type == 'state_action_dict':
            sar = []
            for s, ar in self.reward_dict.items():
                ar = tuple(sorted(ar.items()))
                sar.append((s, ar))
            sar = tuple(sorted(sar))
            myhash.extend([sar,])
        else:
            myhash.extend([False,])

        if self.type == 'state_action_nextstate_dict':
            sansr = []
            for s, ansr in self.reward_dict.items():
                ansr_ = []
                for a, nsr in ansr.items():
                    nsr = tuple(sorted(nsr.items()))
                    ansr_.append((a, nsr))
                ansr_ = tuple(sorted(ansr_))
                sansr.append(ansr_)
            sansr = tuple(sorted(sansr))
            myhash.extend([sansr,])
        else:
            myhash.extend([False,])

        if self.type == 'state_feature_based':
            myhash.extend([
                tuple(sorted(self.state_features.items())),
                tuple(sorted(self.feature_rewards.items()))
            ])
        else:
            myhash.extend([False,
                           False])

        self.hash = hash(tuple(myhash))

        return hash(tuple(myhash))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False
