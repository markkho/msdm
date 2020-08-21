from copy import deepcopy
from functools import lru_cache

from .mdp import MDP

from pyrlap_old.algorithms.valueiteration import ValueIteration
from pyrlap_old.core.util import sample_prob_dict

class DictionaryMDP(MDP):
    TERMINAL_ACTION = 'terminal-action'
    TERMINAL_STATE = 'terminal-state'

    def __init__(self,
                 transition_dict,
                 reward_dict,
                 init_state=None,
                 init_states=None,
                 init_state_dist=None,
                 step_cost=0
                 ):
        """
        Parameters
        ==========
        transition_dict : nested dictionary {s: {a: {ns: p}}}
        reward_dict: nested dictionary {s: {a: {ns: r}}}
        init_state: one of transition_dict's keys
        step_cost: a constant applied to all states except terminal actions/states
        """
        self.transition_dict = transition_dict
        self.reward_dict = {}
        for s, a_ns_p in transition_dict.items():
            s_rdict = deepcopy(reward_dict.get(s, {}))
            for a, ns_p in a_ns_p.items():
                s_rdict[a] = deepcopy(s_rdict.get(a, {}))
                for ns, p in ns_p.items():
                    if a == self.__class__.TERMINAL_ACTION or \
                            ns == self.__class__.TERMINAL_STATE:
                        r = 0
                    else:
                        r = s_rdict[a].get(ns, 0) + step_cost
                    s_rdict[a][ns] = r
            self.reward_dict[s] = s_rdict

        if init_state_dist is not None:
            self.init_state_dist = init_state_dist
        elif init_states is not None:
            self.init_state_dist = {s: 1/len(init_states) for s in init_states}
        elif init_state is not None:
            self.init_state_dist = {init_state: 1}

        self.step_cost = step_cost

        self.states = list(self.transition_dict.keys())

    def get_init_state(self):
        return sample_prob_dict(self.init_state_dist)

    def get_init_states(self):
        return list(self.init_state_dist.keys())

    def get_init_state_dist(self):
        return self.init_state_dist

    def is_absorbing(self, s):
        return False

    def is_terminal(self, s):
        return s == self.__class__.TERMINAL_STATE

    def is_terminal_action(self, a):
        return a == self.__class__.TERMINAL_ACTION

    def transition_reward_dist(self, s, a):
        if self.is_terminal(s):
            return {(self.__class__.TERMINAL_STATE, 0): 1}
        tdist = self.transition_dict[s][a]
        rdist = self.reward_dict[s][a]
        trdist = {}
        for ns, tprob in tdist.items():
            trdist[(ns, rdist[ns])] = tprob
        return trdist

    def reward(self, s=None, a=None, ns=None):
        return self.reward_dict.get(s, {}).get(a, {}).get(ns, self.step_cost)

    @lru_cache()
    def available_actions(self, s=None):
        if s is None:
            aa = set([])
            for s in self.get_states():
                if s == DictionaryMDP.TERMINAL_STATE:
                    aa.add(DictionaryMDP.TERMINAL_ACTION)
                    continue
                aa = aa.union(self.transition_dict[s].keys())
            aa = sorted(list(aa))
            return aa
        if self.is_terminal(s):
            return [self.__class__.TERMINAL_ACTION, ]
        return list(self.transition_dict[s].keys())

    def get_states(self):
        return self.states

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
