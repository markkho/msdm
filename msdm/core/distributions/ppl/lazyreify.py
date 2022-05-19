import functools
import ast
import inspect
from msdm.core.utils.funcutils import method_cache
from msdm.core.distributions.distributions import FiniteDistribution
from msdm.core.distributions.ppl.reify import FunctionReifier

class LazilyReifiedDistribution(FiniteDistribution):
    def __init__(self, function, reified_function, args, kwargs):
        assert reified_function._original_function == function
        self._function = function
        self._reified_function = reified_function
        self.args = args
        self.kwargs = kwargs
        self._reified = False

    @method_cache
    def _reified_dist(self):
        self._reified = True
        return self._reified_function(*self.args, **self.kwargs)

    def sample(self, *, rng=None):
        if self._reified:
            return self._reified_dist().sample(rng=rng)
        if rng is not None:
            # for this to work, the function needs to take rng as a variable
            return self._function(*self.args, **self.kwargs, rng=rng)
        return self._function(*self.args, **self.kwargs)

    def prob(self, e):
        return self._reified_dist().prob(e)

    def __len__(self):
        return self._reified_dist().__len__()

    @property
    def support(self):
        return self._reified_dist().support

    __repr__ = object.__repr__

class LazyFunctionReifier(FunctionReifier):
    """
    Handles creating a reified function that
    returns `LazilyReifiedDistribution`s.
    """
    def __init__(self, function, check_factor_statements=True):
        FunctionReifier.__init__(self, function)
        if check_factor_statements:
            FactorCheck().visit(self.func_ast)
        self._normal_reified_func = self.reified_function
        @functools.wraps(self._normal_reified_func)
        def wrapped(*args, **kwargs):
            return LazilyReifiedDistribution(
                self.function,
                self._normal_reified_func,
                args, kwargs
            )
        wrapped._original_function = self._normal_reified_func._original_function
        self.reified_function = wrapped

class FactorCheck(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "factor":
            raise SyntaxError(
                "Lazy sampling with factor statements is disallowed. "+
                "If you want to ignore factor statements, set "+
                "`check_factor_statements=False`"
            )
        self.generic_visit(node)
