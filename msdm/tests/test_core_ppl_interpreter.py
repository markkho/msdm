import ast
import math
from collections import defaultdict
from types import SimpleNamespace
from collections import Counter
import textwrap
from msdm.core.distributions.ppl.interpreter import Interpreter, \
    Context, factor
from msdm.core.distributions.ppl.utils import strip_comments
from msdm.core.distributions import DictDistribution

def uniform_w_repeat(elements):
    return DictDistribution({
        e: c/len(elements) for e, c in Counter(elements).items()
    })

def flip(p):
    return DictDistribution({True: p, False: 1 - p})

def clean_code(code):
    return textwrap.dedent(strip_comments(code))

def _compare_deterministic_output(code, exp_values, init_context=None):
    code = clean_code(code)
    node = ast.parse(code)
    if init_context is None:
        init_context = Context({}, {}, 0, None)
    contexts = list(Interpreter().run(node, context=init_context))
    for name, val in exp_values.items():
        assert contexts[0].context[name] == val
    return [c.context for c in contexts]

def _compare_stochastic_outcome(code, varlist, exp_dist, init_vars):
    code = clean_code(code)
    node = ast.parse(code)
    init_context = Context({**init_vars}, {}, 0, None)
    contexts = list(Interpreter().run(node, context=init_context))

    res_dist = defaultdict(lambda: 0)
    for c in contexts:
        assn = tuple([c.context[vname] for vname in varlist])
        res_dist[assn] += math.exp(c.score)
    res_dist = DictDistribution(res_dist)
    assert res_dist.isclose(exp_dist), f"exp: {exp_dist}, res: {res_dist}"
    return res_dist

def test_deterministic_runs():
    _compare_deterministic_output(
        code="""
            x = 1
            y = 2
            res = x + y
            res += 1
        """,
        exp_values={
            'res': 4
        }
    )
    res = _compare_deterministic_output(
        code="""
            a = 'A'
            b = 'A' + 'B'
            c = abc(a)
            d = thing.add_D('A')
            e = SimpleNamespace(attribute='E')
            f = "A" + e.attribute
            if True:
                g = "G"
            if False:
                g = "not reachable"
            h = 'H' if True else 'not reachable'
        """,
        exp_values={
            'a': 'A',
            'b': 'AB',
            'c': 'Aabc',
            'd': 'AD',
            'e': SimpleNamespace(attribute='E'),
            'f': 'AE',
            'g': 'G',
            'h': 'H'
        },
        init_context=Context(
            context=dict(
                abc = lambda s: s + 'abc',
                thing = SimpleNamespace(add_D=lambda s: s+'D'),
                SimpleNamespace = SimpleNamespace
            ),
            global_context={},
            score=0,
            status=None
        )
    )

def test_Return():
    _compare_deterministic_output(
        code="return 154",
        exp_values={Interpreter.RETURN_VAR_NAME: 154}
    )


def test_Distribution_sample_method_call():
    # call with Distribution.sample method
    _compare_stochastic_outcome(
        code="""
        x = P.sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): .25,
            ('b',): .75
        }),
        init_vars=dict(P=DictDistribution(a=.25, b=.75))
    )

    # call with non-Distribution.sample method
    _compare_stochastic_outcome(
        code="""
        x = P.sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): 1,
        }),
        init_vars=dict(P=SimpleNamespace(sample = lambda : 'a'))
    )

    # call with detached Distribution.sample method
    _compare_stochastic_outcome(
        code="""
        f = P.sample
        x = f()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): .25,
            ('b',): .75
        }),
        init_vars=dict(P=DictDistribution(a=.25, b=.75))
    )

def test_UnaryOp_inverse_operator():
    _compare_stochastic_outcome(
        code="""
        x = ~P
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): .25,
            ('b',): .75
        }),
        init_vars=dict(P=DictDistribution(a=.25, b=.75))
    )

    _compare_stochastic_outcome(
        code="""
        x = ~P
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            (~10,): 1
        }),
        init_vars=dict(P=10)
    )

    _compare_stochastic_outcome(
        code="""
        x = -P
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            (-10,): 1
        }),
        init_vars=dict(P=10)
    )

def test_factor_statement():
    res = _compare_stochastic_outcome(
        code="""
        x = P.sample()
        factor({'a': .25, 'b': .75}[x])
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): .25*.5,
            ('b',): .75*.5
        }),
        init_vars=dict(
            P=DictDistribution(a=.5, b=.5),
            factor=factor
        )
    )
    res = _compare_stochastic_outcome(
        code="""
        x = uniform_w_repeat('ab').sample()
        y = uniform_w_repeat('abb').sample()
        factor(x == y)
        """,
        varlist=('x', 'y'),
        exp_dist=DictDistribution({
            ('a', 'a'): .5*(1/3),
            ('b', 'b'): .5*(2/3)
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat,
            factor=factor
        )
    )

def test_If_statement():
    res = _compare_stochastic_outcome(
        code="""
        if uniform_w_repeat([True, False, False]).sample():
            x = uniform_w_repeat('ab').sample()
        else:
            x = uniform_w_repeat('ccd').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(2/3),
            ('d',): (2/3)*(1/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )
    res = _compare_stochastic_outcome(
        code="""
        if uniform_w_repeat([True, False, False]).sample():
            x = uniform_w_repeat('ab').sample()
        else:
            x = uniform_w_repeat('ccd').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(2/3),
            ('d',): (2/3)*(1/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )
    res = _compare_stochastic_outcome(
        code="""
        if uniform_w_repeat([True, False, False]).sample():
            x = uniform_w_repeat('ab').sample()
        else:
            if uniform_w_repeat([True, False, False]).sample():
                x = uniform_w_repeat('ccd').sample()
            else:
                x = uniform_w_repeat('eff').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(1/3)*(2/3),
            ('d',): (2/3)*(1/3)*(1/3),
            ('e',): (2/3)*(2/3)*(1/3),
            ('f',): (2/3)*(2/3)*(2/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )
    res = _compare_stochastic_outcome(
        code="""
        if uniform_w_repeat([True, False, False]).sample():
            x = uniform_w_repeat('ab').sample()
        elif uniform_w_repeat([True, False, False]).sample():
            x = uniform_w_repeat('ccd').sample()
        else:
            x = uniform_w_repeat('eff').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(1/3)*(2/3),
            ('d',): (2/3)*(1/3)*(1/3),
            ('e',): (2/3)*(2/3)*(1/3),
            ('f',): (2/3)*(2/3)*(2/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )

def test_IfExp():
    res = _compare_stochastic_outcome(
        code="""
        cond_dist = uniform_w_repeat([True, False, False])
        x = uniform_w_repeat('ab').sample() if cond_dist.sample() else uniform_w_repeat('ccd').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(2/3),
            ('d',): (2/3)*(1/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )
    res = _compare_stochastic_outcome(
        code="""
        cond_dist = uniform_w_repeat([True, False, False])
        x = uniform_w_repeat('ab').sample() if \
            cond_dist.sample() else \
                uniform_w_repeat('ccd').sample() if \
                cond_dist.sample() else \
                uniform_w_repeat('eff').sample()
        """,
        varlist=('x',),
        exp_dist=DictDistribution({
            ('a',): (1/3)*.5,
            ('b',): (1/3)*.5,
            ('c',): (2/3)*(1/3)*(2/3),
            ('d',): (2/3)*(1/3)*(1/3),
            ('e',): (2/3)*(2/3)*(1/3),
            ('f',): (2/3)*(2/3)*(2/3),
        }),
        init_vars=dict(
            uniform_w_repeat=uniform_w_repeat
        )
    )
