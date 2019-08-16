from collections import Counter

from .tiger import TigerProblem
from pyrlap.core.pomdp import POMDPAgent


class TigerCounterAgent(POMDPAgent):
    def __init__(self, tiger: TigerProblem, count_diff=2):
        super().__init__(tiger)
        self.rc = Counter()
        self.cd = count_diff

    def initialize(self, internal_state=None):
        self.rc = Counter()

    def act(self, **kwargs):
        if (self.rc['left-roar']) == (self.rc['right-roar'] + self.cd):
            return 'right-door'
        elif (self.rc['right-roar']) == (self.rc['left-roar'] + self.cd):
            return 'left-door'
        return 'listen'

    def update(self, o: "observation"):
        if o != "reset":
            self.rc[o] += 1
        else:
            self.rc = Counter()

    def get_internal_state(self):
        return self.rc['left-roar'], self.rc['right-roar']