import inspect
import functools
from collections import defaultdict
import textwrap
from msdm.core.distributions.jointprobabilitytable import \
    Assignment, JointProbabilityTable
from msdm.core.distributions.factors import \
    combine, make_factor, InconsistentVariablesException

def test_combine_factors():
    # todo: test debugging and combining
    @make_factor()
    def f0() -> ['x', 'y']:
        return JointProbabilityTable.from_pairs([
            [dict(x=1, y=3), .5],
            [dict(x=6, y=3), .5],
        ])

    @make_factor()
    def f1(x, y) -> ['z']:
        if x > y:
            return JointProbabilityTable.from_pairs([
                [dict(z=y+x), .7],
                [dict(z=y), .1],
                [dict(z=x), .2]
            ])
        return JointProbabilityTable.from_pairs([
            [dict(z=y-x), .7],
            [dict(z=x), .1],
            [dict(z=y), .2]
        ])

    @make_factor()
    def f2() -> ['z2']:
        return JointProbabilityTable.from_pairs([
            [dict(z2=10), .5],
            [dict(z2=11), .5],
        ])

    @make_factor()
    def f3() -> ['z']:
        return JointProbabilityTable.from_pairs([
            [dict(z=10), .5],
            [dict(z=11), .5],
        ])

    assert f1(3, 6).join(f2).join(f3) == combine([f1, f2, f3])(3, 6)
    assert f0().join(f1) == combine([f0, f1])()
    assert combine([f0, f1, f2])() == f0().join(f1).join(f2)
    assert combine([f0, f1])().join(f2) == combine([f0, f1, f2])()

def test_factor_debug_mode():
    def f() -> ['z']:
        return JointProbabilityTable.from_pairs([
            [dict(z=10), .5],
            [dict(y=11), .5],
        ])
    def g() -> ['z']:
        return 100
    f_miss = make_factor(debug_mode=False)(f)
    f_catch = make_factor(debug_mode=True)(f)
    g_miss = make_factor(debug_mode=False)(g)
    g_catch = make_factor(debug_mode=True)(g)

    f_miss()
    try:
        f_catch()
        assert False
    except InconsistentVariablesException:
        pass

    g_miss()
    try:
        g_catch()
        assert False
    except ValueError:
        pass

def test_combine_tables_and_factors():
    @make_factor()
    def f0() -> ['x', 'y']:
        return JointProbabilityTable.from_pairs([
            [dict(x=1, y=3), .2],
            [dict(x=6, y=3), .8],
        ])

    @make_factor()
    def f1(x, y) -> ['z']:
        if x > y:
            return JointProbabilityTable.from_pairs([
                [dict(z=y+x), .7],
                [dict(z=y), .1],
                [dict(z=x), .2]
            ])
        return JointProbabilityTable.from_pairs([
            [dict(z=y-x), .7],
            [dict(z=x), .1],
            [dict(z=y), .2]
        ])
    f1_table = JointProbabilityTable.from_pairs(
        [(Assignment.from_kwargs(x=1, y=3)+assn, p) for assn, p in f1(1, 3).items()] +
        [(Assignment.from_kwargs(x=6, y=3)+assn, p) for assn, p in f1(6, 3).items()]
    )
    # it shouldn't be necessary to normalize these
    from_tables = f0().join(f1_table)
    from_combine = combine([f0, f1])()
    from_table_combine1 = combine([f0(), f1_table])()
    from_table_combine2 = combine([f0, f1_table])()
    from_table_combine3 = combine([f0(), f1])()
    assert from_combine.isclose(from_tables)
    assert from_combine.isclose(from_table_combine1)
    assert from_combine.isclose(from_table_combine2)
    assert from_combine.isclose(from_table_combine3)
