from itertools import product

from pyrlap_old.core.pomdp import PartiallyObservableMarkovDecisionProcess
from pyrlap_old.core.mdp import MDP
from pyrlap_old.core.util import sample_prob_dict

class LoadUnloadMDP(MDP):
    def __init__(self,
                 length=8,
                 unload_reward=10):
        self.length = length
        self.unload_reward = unload_reward

    def get_states(self):
        return list(product(list(range(self.length)), ['load', 'no-load']))

    def get_init_state_dist(self):
        return {(0, 'no-load'): 1.0}

    def get_init_state(self):
        return (0, 'no-load')

    def is_terminal(self, s):
        return False

    def is_absorbing(self, s):
        return False

    def available_actions(self, s=None):
        return [-1, 1]

    def transition_dist(self, s, a):
        loc, load = s
        next_loc = max(0, min(self.length - 1, loc + a))
        if next_loc == (self.length - 1):
            next_load = 'load'
        elif next_loc == 0:
            next_load = 'no-load'
        else:
            next_load = load

        return {(next_loc, next_load): 1.0}

    def transition(self, s, a):
        return sample_prob_dict(self.transition_dist(s, a))

    def reward_dist(self, s=None, a=None, ns=None):
        loc, load = s
        nloc, nload = ns
        if load == 'load' and nload == 'no-load':
            return {self.unload_reward: 1.0}
        else:
            return {0: 1.0}

    def reward(self, s=None, a=None, ns=None):
        return sample_prob_dict(self.reward_dist(s, a, ns))

class LoadUnload(PartiallyObservableMarkovDecisionProcess):
    def __init__(self,
                 length=8,
                 unload_reward=10,
                 ):
        mdp = LoadUnloadMDP(length, unload_reward)
        super().__init__(mdp)

    def observation_dist(self, a, ns):
        nloc, nload = ns
        if nloc == 0 or nloc == (self.mdp.length - 1):
            return {nload: 1.0}
        else:
            return {'none': 1.0}