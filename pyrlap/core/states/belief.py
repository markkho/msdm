from functools import total_ordering
from .state import State

@total_ordering
class Belief(State):
    def __init__(self, bdict, owner='', **kwargs):
        super(Belief, self).__init__(
            bdict,
            immutable=True,
            **kwargs
            )
        self.owner = owner