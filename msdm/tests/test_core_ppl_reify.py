from msdm.core.distributions.ppl.reify import FunctionReifier, ReifiedFunctionRunningError
from msdm.core.distributions import DictDistribution
from msdm.core.distributions.ppl.interpreter import factor

def reify(function):
    return FunctionReifier(function).reified_function

def flip(p):
    return DictDistribution({True: p, False: 1 - p})

def test_recursive_error_handling():
    @reify
    def f(n, p=.5):
        if n == 0:
            return 0
        if ~flip(p):
            return 0
        return n + f(n - 1)

    try:
        f(3)
        assert False
    except ReifiedFunctionRunningError:
        pass


    # This is an example of how it can be written to not
    # throw the error
    def f(n, p=.5):
        @reify
        def _f(n):
            if n == 0:
                return 0
            if ~flip(p):
                return 0
            return 1 + ~f(n - 1)
        return _f(n)

    f(3)
