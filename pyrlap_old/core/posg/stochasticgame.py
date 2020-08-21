from typing import Union, Iterable, Mapping

from pyrlap_old.core.util import sample_prob_dict

class StochasticGame(object):
    """
    """
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def get_init_state(self):
        raise NotImplementedError

    def get_init_state_dist(self):
        raise NotImplementedError

    def get_init_states(self):
        return NotImplementedError

    def is_terminal(self, s: "state"):
        raise NotImplementedError

    def is_absorbing(self, s: "state"):
        raise NotImplementedError

    def is_any_terminal(self, s: "state"):
        raise NotImplementedError

    def is_terminal_action(self, ja: "joint action"):
        raise NotImplementedError

    def get_terminal_states(self):
        raise NotImplementedError

    def transition_reward_dist(self,
                               s: "state",
                               ja: "joint action"):
        raise NotImplementedError

    def transition_dist(self,
                        s: "state",
                        ja: "joint action"):
        trdist = self.transition_reward_dist(s, ja)
        tdist = {}
        for (ns, r), p in trdist.items():
            tdist[ns] = tdist.get(ns, 0) + p
        return tdist

    def transition(self,
                   s: "state",
                   ja: "joint action"):
        return sample_prob_dict(self.transition_reward_dist(s, ja))[0]

    def reward(self,
               s: "state"=None,
               ja: "joint action"=None,
               ns: "next state"=None):
        raise NotImplementedError

    def reward_dist(self,
                    s: "state"=None,
                    ja: "joint action"=None,
                    ns: "next state"=None):
        raise NotImplementedError

    def available_actions(self,
                          s: "state" =None):
        raise NotImplementedError