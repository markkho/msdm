from random import randint, seed

from .pomdp import PartiallyObservableMarkovDecisionProcess
from .pomdpagent import POMDPAgent

class RandomPOMDPAgent(POMDPAgent):
    def __init__(self,
                 pomdp: PartiallyObservableMarkovDecisionProcess,
                 rand_seed=0):
        super().__init__(pomdp)
        seed(rand_seed)

    def initialize(self, internal_state=None):
        pass

    def act(self, **kwargs):
        aa = self.pomdp.available_actions()
        return aa[randint(0, len(aa) - 1)]

    def update(self, o: "observation"):
        oo = self.pomdp.get_observations()
        pass

    def get_internal_state(self):
        return 0