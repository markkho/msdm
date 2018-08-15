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
                 **kwargs):
        if fvals is None:
            fvals = kwargs
        super(State, self).__init__(fvals)
        self.features = tuple(sorted(list(fvals.keys())))
        self.vals = tuple([self[f] for f in self.features])

        if variables is None:
            variables = {}
        self.variables = variables
        # self.constructing = True
        self.__consolidate_vars()

        self.var_order = tuple(sorted(list(self.variables.keys())))
        self.var_vals = tuple([self.variables[v] for v in self.var_order])
        self.immutable = immutable

        self.__calc_hash()
        # self.constructing = False

    def __get_variable_value(self, s):
        if not is_variable(s):
            return s

        # Needed to handle case when state is being constructed
        # but variables are not accessible yet
        if s not in self.variables:
            return s

        return self.__get_variable_value(self.variables[s])

    def __consolidate_vars(self):
        """
        Updates self.variables to have the same variable values
        as all internal states, checking for conflicts.
        :return:
        """
        def __findstore_vars(s: State, parent : State = None):
            if is_variable(s):
                s = parent.__get_variable_value(s)

                # if the value isn't accessible yet
                if is_variable(s):
                    return

            if not isinstance(s, State):
                return

            # ensure variables are the same
            for var in s.variables:
                if var in self.variables and \
                        self.variables[var] != s.variables[var]:
                    raise ValueError(
                        "Different variable {} in sub-states".format(var))

            for varval in s.variables.values():
                __findstore_vars(varval, s)

            for f, v in zip(s.features, s.vals):
                __findstore_vars(f, s)
                __findstore_vars(v, s)

            self.variables.update(s.variables)
            # s.variables = self.variables

        for f, v in self.items():
            __findstore_vars(f, self)
            __findstore_vars(v, self)

    def __calc_hash(self):
        def _rec_hash(s: State):
            if is_variable(s):
                s = self.__get_variable_value(s)

                # if value isn't accessible yet
                if is_variable(s):
                    return hash(s)

                return _rec_hash(s)

            if not isinstance(s, State):
                return hash(s)
            fv_hashes = []
            for f, v in s.items():
                fv_hashes.append((_rec_hash(f), _rec_hash(v)))
            return hash(tuple(fv_hashes))
        self._hash = _rec_hash(self)

    def __hash__(self):
        return self._hash

    def items(self):
        for f in self: yield (f, self[f])

    def keys(self):
        for f in self: yield f

    def values(self):
        for f in self: yield self[f]

    def __myrepr__(self,
                   show_vars=True,
                   show_nested_vars=False,
                   element_newlines=False):
        fvals = []
        for f, v in self.items():
            if isinstance(f, State):
                frep = f.__myrepr__(show_vars=show_nested_vars,
                                    show_nested_vars=show_nested_vars)
            else:
                frep = repr(f)
            if isinstance(v, State):
                vrep = v.__myrepr__(show_vars=show_nested_vars,
                                    show_nested_vars=show_nested_vars)
            else:
                vrep = repr(v)
            fvals.append('{}: {}'.format(frep, vrep))
        if element_newlines:
            fvals = ',\n'.join(fvals+[''])
        else:
            fvals = ', '.join(fvals)

        toreturn = [self.__class__.__name__+'({', fvals, '}']

        if show_vars:
            varvals = []
            for var, val in zip(self.var_order, self.var_vals):
                if isinstance(var, State):
                    varrep = \
                        var.__myrepr__(show_vars=show_nested_vars,
                                          show_nested_vars=show_nested_vars)
                else:
                    varrep = repr(var)

                if isinstance(val, State):
                    valrep = \
                        val.__myrepr__(show_vars=show_nested_vars,
                                       show_nested_vars=show_nested_vars)
                else:
                    valrep = repr(val)
                varvals.append('{}: {}'.format(varrep, valrep))
            if element_newlines:
                varvals = ',\n'.join(varvals+[''])
            else:
                varvals = ', '.join(varvals)
            toreturn.extend([', variables={', varvals, '}'])

        toreturn[-1] = toreturn[-1]+')'
        return ''.join(toreturn)

    def __repr__(self):
        return self.__myrepr__()

    def __setitem__(self, key, value):
        if self.immutable:
            raise TypeError("State is immutable")
        super(State, self).__setitem__(key, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

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

    def clear_variables(self):
        self.variables = {}
        self.var_order = ()
        self.var_vals = ()

    def pretty_str(self, indent='    ', maxcols=80):
        def __recursion(s, depth):
            # todo: flat_chars is counting nested variables
            flat_chars = len(repr(s)) + len(indent * (depth + 1))
            if not isinstance(s, State) or flat_chars < maxcols:
                stack.append(repr(s))
                return

            indents = indent * (depth + 1)
            stack.append("{}({{\n".format(s.__class__.__name__))

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
            if depth > 0:
                stack.append(")")

        stack = []

        __recursion(self, 0)

        depth = 0
        for vi, (var, val) in enumerate(zip(self.var_order, self.var_vals)):
            if vi == 0:
                stack.append(",\n" + (indent * depth) + "variables={\n")
            __recursion(var, depth=depth + 1)
            stack.append(": ")
            __recursion(val, depth=depth + 1)
            if vi == (len(self.var_order) - 1):
                stack.append("\n")
            else:
                stack.append(",\n")
        if len(self.var_order) > 0:
            stack.append("{}}}".format(indent * depth))
        stack.append(")")
        return ''.join(stack)

    def freeze(self):
        def __recursion(s):
            if is_variable(s):
                return __recursion(self.variables[s])

            if not isinstance(s, State):
                return s

            fvals = {}
            for f in s.features:
                v = s[f]
                fvals[__recursion(f)] = __recursion(v)
            return s.__class__(fvals)

        return __recursion(self)

    def vectorize(self):
        raise NotImplementedError