from copy import deepcopy
from functools import total_ordering

def is_variable(x):
    return isinstance(x, str) and x[0] == '_'

@total_ordering
class State(dict):
    def __init__(self,
                 fvals=None,
                 variables=None,
                 immutable=True,

                 # _prefixstr=None,
                 # _openstr='({',
                 # _closestr='})',
                 **kwargs):
        if fvals is None:
            fvals = kwargs
        super(State, self).__init__(fvals)
        self.features = tuple(sorted(list(fvals.keys())))
        self.vals = tuple([self[f] for f in self.features])
        if variables is None:
            variables = {}
        self.variables = variables
        self.var_order = tuple(sorted(list(self.variables.keys())))
        self.var_vals = tuple([self.variables[v] for v in self.var_order])
        self.immutable = immutable

        # if _prefixstr is None:
        #     _prefixstr = self.__class__.__name__
        # self._prefixstr = _prefixstr
        # self._openstr = _openstr
        # self._closestr = _closestr

    def __hash__(self):
        return hash((self.features, self.vals, self.var_order, self.var_vals))

    def __repr__(self):
        fvals = ['{}: {}'.format(repr(f), repr(v)) for f, v in self.items()]
        fvals = ', '.join(fvals)
        if len(self.var_order) == 0:
            return '{}({{{}}})'.format(self.__class__.__name__, fvals)
        else:
            varvals = zip(self.var_order, self.var_vals)
            varvals = ['{}: {}'.format(repr(r), repr(l)) for r, l in varvals]
            varvals = ', '.join(varvals)

            return '{class}({{{fvals}}}, variables={{{varvals}}})'.format(**{
                'class': self.__class__.__name__,
                'fvals': fvals,
                'varvals': varvals
            })

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

    def __str__(self):
        return self.pretty_str()

    def pretty_str(self, indent='    ', maxcols=80):
        def __recursion(s, depth):
            flat_chars = len(repr(s)) + len(indent * (depth + 1))
            if not isinstance(s, State) or flat_chars < maxcols:
                stack.append(repr(s))
                return

            indents = indent * (depth + 1)
            stack.append("{}({{\n".format(self.__class__.__name__))
            # stack.append(s._prefixstr+s._openstr+'\n')

            for fi, (f, v) in enumerate(s.items()):
                stack.append(indents)
                __recursion(f, depth=depth + 1)
                stack.append(': ')
                __recursion(v, depth=depth + 1)
                if fi == (len(s) - 1):
                    stack.append('\n')
                else:
                    stack.append(',\n')
            stack.append("{}}}".format(indent * depth))

            for vi, (var, val) in enumerate(zip(s.var_order, s.var_vals)):
                if vi == 0:
                    stack.append(",\n"+(indent * depth)+"variables={\n")
                stack.append(indents)
                __recursion(var, depth=depth + 1)
                stack.append(": ")
                __recursion(val, depth=depth + 1)
                if vi == (len(s.var_order) - 1):
                    stack.append("\n")
                else:
                    stack.append(",\n")
            if len(s.var_order) > 0:
                stack.append("{}}})".format(indent * depth))

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

    def vectorize(self):
        raise NotImplementedError