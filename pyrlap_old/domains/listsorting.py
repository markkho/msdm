from functools import lru_cache
from itertools import permutations, combinations, product
from random import shuffle

from pyrlap_old.core.pomdp import PartiallyObservableMarkovDecisionProcess
from pyrlap_old.core.mdp import MDP
from pyrlap_old.core.util import sample_prob_dict

class ListSortingTaskMDP(MDP):
    def __init__(self,
                 list_size=6,
                 initial_state_type="uniform",
                 sort_reward=100,
                 comparison_cost=-1
                 ):
        self.list_size = list_size
        self.initial_state_type = initial_state_type
        self.terminal_slist = tuple([-1, ]*list_size)
        self.sort_reward = sort_reward
        self.comparison_cost = comparison_cost
        self.wait_action = (-1, -1)

    def get_states(self):
        """
        Because of how the POMDP obs. func interface works,
        states need to store information about if there was
        previously a successful swap.
        """
        sslist = list(permutations(range(self.list_size), self.list_size))
        sslist = [tuple(slist) for slist in sslist] + [self.terminal_slist, ]
        ss = list(product(sslist, ['swap', 'no-swap', 'init']))
        return ss

    @lru_cache()
    def get_init_state_dist(self):
        if self.initial_state_type == 'uniform':
            ss = self.get_states()
            ss0 = [s for s in ss if s[1] == 'init']
            s0 = {s: 1/len(ss0) for s in ss0}
            return s0
        else:
            raise Exception("Unknown initial state type")

    def get_init_state(self):
        return sample_prob_dict(self.get_init_state_dist())

    def is_sorted(self, slist):
        return all([slist[i] < slist[i+1] for i in range(len(slist) - 1)])

    def is_terminal(self, s):
        slist, swap = s
        return slist == self.terminal_slist

    def is_absorbing(self, s):
        slist, swap = s
        return self.is_sorted(slist)

    @lru_cache()
    def available_actions(self, s=None):
        aa = combinations(list(range(self.list_size)), 2)
        aa = [tuple(a) for a in aa]
        aa.append(self.wait_action)
        return aa

    def transition_dist(self, s, a):
        slist, swap = s
        if self.is_sorted(slist):
            nsswap = "no-swap"
            return {(self.terminal_slist, nsswap): 1.0}

        if a == self.wait_action:
            nsswap = "no-swap"
            return {(slist, nsswap): 1.0}

        i0_i1 = sorted(list(a)) # attempt to swap i1 with i0
        i0, i1 = i0_i1
        if slist[i1] < slist[i0]:
            nslist = list(slist)
            nslist[i0] = slist[i1]
            nslist[i1] = slist[i0]
            nslist = tuple(nslist)
            nsswap = 'swap'
        else:
            nslist = slist
            nsswap = 'no-swap'
        return {(nslist, nsswap): 1.0}

    def transition(self, s, a):
        return sample_prob_dict(self.transition_dist(s, a))

    def reward(self, s=None, a=None, ns=None):
        if self.is_terminal(s):
            return 0

        slist, swap = s
        if self.is_sorted(slist):
            return self.sort_reward

        if a == self.wait_action:
            return 0

        return self.comparison_cost


class ListSortingTask(PartiallyObservableMarkovDecisionProcess):
    def __init__(self,
                 list_size=6,
                 initial_state_type="uniform",
                 sort_reward=100,
                 comparison_cost=-1
                 ):
        mdp = ListSortingTaskMDP(
            list_size=list_size,
            initial_state_type=initial_state_type,
            sort_reward=sort_reward,
            comparison_cost=comparison_cost
        )
        super().__init__(mdp)

    def observation_dist(self, a, ns):
        nslist, nsswap = ns
        return {nsswap: 1.0}
