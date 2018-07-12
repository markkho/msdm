from copy import deepcopy

from .mdp import MDP

from pyrlap.algorithms.valueiteration import ValueIteration

class DictionaryMDP(MDP):
    TERMINAL_ACTION = 'terminal-action'
    TERMINAL_STATE = 'terminal-state'

    def __init__(self,
                 transition_dict,
                 reward_dict,
                 init_state=None,
                 init_states=None,
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

        if init_states is None:
            init_states = []
        if init_state is not None:
            init_states.append(init_state)
        else:
            init_state = init_states[0]
        self.init_state = init_state
        self.init_states = init_states

        self.step_cost = step_cost

        self.states = list(self.transition_dict.keys())
        self.states.append(self.__class__.TERMINAL_STATE)

    def get_init_state(self):
        return self.init_state

    def get_init_states(self):
        return self.init_states

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

    def available_actions(self, s):
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
