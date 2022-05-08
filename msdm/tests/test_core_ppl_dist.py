import numpy as np
import random
import timeit
from msdm.core.distributions import uniform, flip, DictDistribution
from msdm.core.distributions.ppl.reify import FunctionReifier
from msdm.core.distributions.ppl.interpreter import factor
from msdm.core.distributions.ppl.lazyreify import LazyFunctionReifier

def test_argument_extraction():
    def f(a, b, c=10, **kws):
        pass

    # positional, keyword, and extra keyword args
    res = FunctionReifier(f).extract_arg_kws(
        args=(),
        kws={'a': 1, 'b': 2, 'c': 3, 'd':10}
    )
    expected = dict(a=1, b=2, c=3, kws=dict(d=10))
    assert res == expected

    res = FunctionReifier(f).extract_arg_kws(
        args=(1,),
        kws={'b': 2, 'c': 3, 'd':10}
    )
    expected = dict(a=1, b=2, c=3, kws=dict(d=10))
    assert res == expected

    res = FunctionReifier(f).extract_arg_kws(
        args=(1, 2),
        kws={'d':10}
    )
    expected = dict(a=1, b=2, c=10, kws=dict(d=10))
    assert res == expected

    # this prevents positional arguments
    def f(*, a, b, c=10, **kws):
        pass
    res = FunctionReifier(f).extract_arg_kws(
        args=(),
        kws={'a': 1, 'b': 2, 'c': 3, 'd':20}
    )
    expected = dict(a=1, b=2, c=3, kws=dict(d=20))
    assert res == expected

def test_closures():
    def outer():
        a = 10
        b = 100
        def f():
            c = a + 10
        return f
    f = outer()
    res = FunctionReifier(f).closure()
    assert 'a' in res and 'b' not in res

    def outer():
        a = 10
        b = 100
        def f(d=b): # this is known to not work
            c = a + 10
        return f
    f = outer()
    try:
        res = FunctionReifier(f).extract_arg_kws(args=(), kws={})
        raise
    except NameError:
        pass
    res = FunctionReifier(f).closure()
    assert 'a' in res and 'b' not in res

def test_function_reification():
    def f(p=.5):
        cond = ~flip(p)
        if cond:
            return ~uniform('aab')
        else:
            return ~uniform('abb')
    F = FunctionReifier(f).reified_function

    # default value
    assert F().isclose(uniform('ab'))

    # arg and kw
    for p in [0, .1, .5, .6, .9, 1.0, .88823923]:
        exp = DictDistribution({
            'a': p*(2/3)+(1-p)*(1/3),
            'b': p*(1/3)+(1-p)*(2/3)
        })
        assert F(p=p).isclose(exp)
        assert F(p).isclose(exp)

    # chain
    P = DictDistribution(zip(np.linspace(0, 1, 5), range(6)))
    P = P.normalize()
    P.chain(F)

def test_lazy_function_reification():
    def f(p=.5):
        cond = ~flip(p)
        if cond:
            return ~uniform('aab')
        else:
            return ~uniform('abb')
    F = LazyFunctionReifier(f).reified_function

    # default value
    assert F().isclose(uniform('ab'))

    # arg and kw
    for p in [0, .1, .5, .6, .9, 1.0, .88823923]:
        exp = DictDistribution({
            'a': p*(2/3)+(1-p)*(1/3),
            'b': p*(1/3)+(1-p)*(2/3)
        })
        assert F(p=p).isclose(exp)
        assert F(p).isclose(exp)

    # chain
    P = DictDistribution(zip(np.linspace(0, 1, 5), range(6)))
    P = P.normalize()
    P.chain(F)

def test_compare_lazy_eager_reified_functions_timing():
    def f(p):
        return flip(p).sample()
    F_lazy = LazyFunctionReifier(f).reified_function
    F_eager = FunctionReifier(f).reified_function

    eager_time = timeit.timeit(lambda : F_eager(random.random()).sample(), number=100)
    lazy_time = timeit.timeit(lambda : F_lazy(random.random()).sample(), number=100)
    assert eager_time > lazy_time

    eager_time = timeit.timeit(lambda : F_eager(random.random()).prob(1), number=100)
    lazy_time = timeit.timeit(lambda : F_lazy(random.random()).prob(1), number=100)
    assert abs(eager_time - lazy_time) < 1

def test_lazy_reifer_factor_check():
    def f1(a, b):
        a.test()
        factor(a.factor(10))
        return a + b
    try:
        LazyFunctionReifier(f1, check_factor_statements=True)
        raise
    except SyntaxError:
        pass
    LazyFunctionReifier(f1, check_factor_statements=False)

    def f2(a, b):
        a.test()
        a.factor(a.factor(10))
        return a + b
    LazyFunctionReifier(f2, check_factor_statements=True)
