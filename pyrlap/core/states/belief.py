from functools import total_ordering
from .state import State

@total_ordering
class Belief(State):
    def __init__(self, bdict=None, owner='', **kwargs):
        super(Belief, self).__init__(
            bdict,
            immutable=True,
            _prefixstr=None,
            _openstr='({',
            _closestr="}}, owner={})".format(repr(owner)),
            **kwargs)
        self.owner = owner

    def __hash__(self):
        return super(Belief, self).__hash__()

    def __eq__(self, other):
        if self.features != other.features:
            raise TypeError("Beliefs have different supports")
        return self.vals == other.vals

    def __setitem__(self, key, value):
        raise TypeError("Belief is immutable")