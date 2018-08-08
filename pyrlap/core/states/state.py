from copy import deepcopy
from functools import total_ordering
from collections import OrderedDict

@total_ordering
class State(dict):
    def __init__(self,
                 fvals=None,
                 immutable=True,

                 _prefixstr='s',
                 _openstr='{',
                 _closestr='}',
                 **kwargs):
        if fvals is None:
            fvals = kwargs
        super(State, self).__init__(fvals)
        self.features = tuple(sorted(list(fvals.keys())))
        self.vals = tuple([self[f] for f in self.features])
        self.immutable = immutable

        self._prefixstr = _prefixstr
        self._openstr = _openstr
        self._closestr = _closestr

    def __hash__(self):
        return hash(tuple([(f, self[f]) for f in self.features]))

    def __str__(self):
        s = []
        for f in self.features:
            s.append('{}={}'.format(f, self[f]))
        return ''.join([self._prefixstr,
                        self._openstr,
                        ', '.join(s),
                        self._closestr,])

    def __setitem__(self, key, value):
        if self.immutable:
            raise TypeError("State is immutable")
        super(State, self).__setitem__(key, value)

    def __eq__(self, other):
        if self.features != other.features:
            raise TypeError("States have different features")
        return hash(self) == hash(other)

    def __lt__(self, other):
        for f in self.features:
            if self[f] < other[f]:
                return True
            elif self[f] > other[f]:
                return False

    def __iter__(self):
        for f in self.features:
            yield f

    def pretty_str(self, indent='    ', maxcols=80):
        def __recursion(s, depth):
            flat_chars = len(str(s)) + len(indent * (depth + 1))
            if not isinstance(s, State) or flat_chars < maxcols:
                stack.append(str(s))
                return

            indents = indent * (depth + 1)
            stack.append(s._prefixstr+s._openstr+'\n')

            for fi, (f, v) in enumerate(s.items()):
                stack.append(indents)
                __recursion(f, depth=depth + 1)
                stack.append(': ')
                __recursion(v, depth=depth + 1)
                if fi == (len(s) - 1):
                    stack.append('\n')
                else:
                    stack.append(',\n')
            stack.append((indent * depth) + s._closestr)

        stack = []

        __recursion(self, 0)
        return ''.join(stack)

    def items(self):
        for f in self: yield (f, self[f])

    def keys(self):
        for f in self: yield f

    def values(self):
        for f in self: yield self[f]

    def freeze(self):
        frozen = deepcopy(self)
        frozen.immutable = True
        return frozen