import numpy as np
import contextlib
import pytest
from msdm.core.table import Table, ProbabilityTable
from msdm.core.table.tableindex import TableIndex, Field, domaintuple, DomainError, SliceError, \
    MultipleIndexError, IndexSizeError
from msdm.core.distributions import Distribution 

def test_TableIndex():
    fields = [
        Field('a', (0, 1)), 
        Field('b', ("x", "y", "z"))
    ]
    idx = TableIndex(fields=fields)
    assert idx.shape == (2, 3)
    assert idx.field_names == ('a', 'b')
    for fi, f in enumerate(idx.fields):
        assert f.name == fields[fi].name
        assert f.domain == fields[fi].domain
    assert idx.field_domains == ((0, 1), ("x", "y", "z"))
    assert set(idx.product()) == set([
        (0, 'x'),
        (0, 'y'),
        (0, 'z'),
        (1, 'x'),
        (1, 'y'),
        (1, 'z'),
    ])
    # product_dict should return exactly these dictionaries
    exp_dicts = [
        dict(a=0, b='x'),
        dict(a=0, b='y'),
        dict(a=0, b='z'),
        dict(a=1, b='x'),
        dict(a=1, b='y'),
        dict(a=1, b='z'),
    ]
    for d in idx.product_dicts():
        exp_dicts.pop(exp_dicts.index(d))
    assert len(exp_dicts) == 0

def test_TableIndex_to_numpy_array_index_conversion():
    idx = TableIndex(
        field_names=("dim1", "dim2"),
        field_domains=(
            (('a', 'b',), 'a', 'b', 'c'),
            ('a', 'b', 'c', 'd'),
        )
    )
    assert idx._array_index('a') == (1,)
    assert idx._array_index(('a', 'b')) == (0,)
    assert idx._array_index(('c', 'c')) == (3, 2)
    assert idx._array_index(['c', 'c']) == [3, 3], idx._array_index(['c', 'c'])
    assert idx._array_index((['c', 'c'],)) == ([3, 3],), idx._array_index(['c', 'c'])

    # error if we use numpy-like "advanced indexing" on more than one field 
    with pytest.raises(MultipleIndexError):
        idx._array_index((['c'], ['c']))

    # error if we try to select more than the num. of fields
    with pytest.raises(IndexSizeError):
        idx._array_index(('a', 'b', 'c', 'd',))

    # unrecognized field selector (dict, which is not hashable nor a list)
    with pytest.raises(IndexError):
        assert idx._array_index((slice(None),{})) == (slice(None),)
    assert idx._array_index(['a', 'b']) == [1, 2]
    assert idx._array_index(['a', 'b', ('a', 'b')]) == [1, 2, 0]
    assert idx._array_index((['a', 'b'],)) == ([1, 2],)
    assert idx._array_index((['a', 'b'], slice(None))) == ([1, 2], slice(None))
    assert idx._array_index((['a', 'b'], 'a')) == ([1, 2], 0)
    assert idx._array_index((('a', 'b'), 'a')) == (0, 0)
    assert idx._array_index(...) == ..., idx._array_index(...)
    assert idx._array_index(slice(None)) == slice(None)
    assert idx._array_index((...,)) == (...,), idx._array_index((...,))
    assert idx._array_index((slice(None),)) == (slice(None),)

def test_TableIndex_equality():
    a = TableIndex(
        fields=[
            Field('x', tuple((1,2,3)))
        ]
    )
    b = TableIndex(
        fields=[
            Field('x', domaintuple((1,2,3)))
        ]
    )
    assert a == b

def test_tuple_domaintuple_equality():
    a = domaintuple(range(100))
    a2 = domaintuple(a)
    b = domaintuple(range(100))
    c = tuple(range(100))
    assert a is a2
    assert a is not b
    assert a == b
    assert a == c

def test_TableIndex_numpy_array_TableIndex_conversion():
    idx = TableIndex(
        field_names=("dim1", "dim2", "dim3"),
        field_domains=(
            domaintuple((('a', 'b',), 'a', 'b')),
            domaintuple(('a', 'b', 'c', 'd')),
            domaintuple((1, 2, 34, 100)),
        )
    )
    arr = np.arange(np.product(idx.shape)).reshape(idx.shape)

    tests = [
        dict(
            sel=[
                ...,
                (...,),
                slice(None),
                (slice(None),),
                (slice(None), slice(None),),
                (slice(None), slice(None), slice(None)),
                (
                    domaintuple([('a', 'b'), 'a', 'b']),
                    slice(None),
                    slice(None)
                ),
                (
                    slice(None),
                    domaintuple(('a', 'b', 'c', 'd')),
                    slice(None)
                ),
                (
                    slice(None),
                    slice(None),
                    domaintuple((1, 2, 34, 100)),
                ),
                (
                    domaintuple([('a', 'b'), 'a', 'b']),
                    ...
                ),
                (
                    slice(None),
                    domaintuple(('a', 'b', 'c', 'd')),
                    ...
                ),
                (
                    ...,
                    domaintuple(('a', 'b', 'c', 'd')),
                    slice(None),
                ),
                (
                    ...,
                    domaintuple((1, 2, 34, 100)),
                ),
            ],
            exp_idx=idx,
        ),
        dict(
            sel=[
                ['a', 'b'],
            ],
            exp_idx=TableIndex(
                field_names=("dim1", "dim2", "dim3"),
                field_domains=(
                    domaintuple(('a', 'b')),
                    domaintuple(('a', 'b', 'c', 'd')),
                    domaintuple((1, 2, 34, 100)),
                )
            )
        ),
        dict(
            sel=[
                [('a', 'b'),],
            ],
            exp_idx=TableIndex(
                field_names=("dim1", "dim2", "dim3"),
                field_domains=(
                    domaintuple((('a', 'b'),)),
                    domaintuple(('a', 'b', 'c', 'd')),
                    domaintuple((1, 2, 34, 100)),
                )
            )
        ),
        dict(
            sel=[
                ('a',),
                ('a', slice(None)),
                ('a', slice(None), slice(None)),
                ('a', ...),
                ('b',),
                ('b', slice(None)),
                ('b', slice(None), slice(None)),
                ('b', ...),
                (('a', 'b'),),
                (('a', 'b'), slice(None)),
                (('a', 'b'), slice(None), slice(None)),
                (('a', 'b'), ...),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim2", domaintuple(['a', 'b', 'c', 'd'])),
                    Field("dim3", domaintuple((1, 2, 34, 100)))
                ]
            )
        ),
        dict(
            sel=[
                (['a',], slice(None), 1),
                (['a',], ..., 1),
                (domaintuple(['a',]), slice(None), 1),
                (domaintuple(['a',]), ..., 1),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple(['a'])),
                    Field("dim2", domaintuple(['a', 'b', 'c', 'd'])),
                ]
            )
        ),
        dict(
            sel=[
                (['a', 'b', ('a', 'b')], slice(None), 1),
                (['a', 'b', ('a', 'b')], ..., 1),
                (domaintuple(['a', 'b', ('a', 'b')]), slice(None), 1),
                (domaintuple(['a', 'b', ('a', 'b')]), ..., 1),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple(['a', 'b', ('a', 'b')])),
                    Field("dim2", domaintuple(['a', 'b', 'c', 'd'])),
                ]
            )
        ),
        dict(
            sel=[
                (['a', ('a', 'b')], ...),
                (domaintuple(['a', ('a', 'b')]), ...),
                (['a', ('a', 'b')], slice(None), slice(None)),
                (domaintuple(['a', ('a', 'b')]), slice(None), slice(None)),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple(['a', ("a", "b")])),
                    Field("dim2", domaintuple(['a', 'b', 'c', 'd'])),
                    Field("dim3", domaintuple((1, 2, 34, 100)))
                ]
            )
        ),
        dict(
            sel=[
                (slice(None), "b"),
                (slice(None), "b", slice(None)),
                (..., "b", slice(None)),
                (slice(None), "b", ...),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple((('a', 'b',), 'a', 'b'))),
                    Field("dim3", domaintuple((1, 2, 34, 100)))
                ]
            )
        ),
        dict(
            sel=[
                (slice(None), ["b",]),
                (slice(None), ["b",], slice(None)),
                (slice(None), ["b",], ...),
                (..., ["b",], slice(None)),
                (slice(None), domaintuple(["b",])),
                (slice(None), domaintuple(["b",]), slice(None)),
                (slice(None), domaintuple(["b",]), ...),
                (..., domaintuple(["b",]), slice(None)),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple((('a', 'b',), 'a', 'b'))),
                    Field("dim2", domaintuple(['b',])),
                    Field("dim3", domaintuple((1, 2, 34, 100)))
                ]
            )
        ),
        dict(
            sel=[
                (slice(None), slice(None), 34,),
                (..., 34,),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple((('a', 'b',), 'a', 'b'))),
                    Field("dim2", domaintuple(('a', 'b', 'c', 'd'))),
                ]
            )
        ),
        dict(
            sel=[
                (slice(None), slice(None), [34,]),
                (..., domaintuple((34,))),
                (slice(None), slice(None), [34,]),
                (..., domaintuple((34,))),
            ],
            exp_idx=TableIndex(
                fields=[
                    Field("dim1", domaintuple((('a', 'b',), 'a', 'b'))),
                    Field("dim2", domaintuple(('a', 'b', 'c', 'd'))),
                    Field("dim3", domaintuple((34,)))
                ]
            )
        ),
        # # Errors expected for weird inputs
        dict(
            sel=[
                (slice(None), ["b"], [100]),
            ],
            error=MultipleIndexError
        ),
        dict(
            sel=[
                (slice(None), "b", slice(2)),
            ],
            error=SliceError
        ),
        dict(
            sel=[
                (..., ("b",)),
            ],
            error=DomainError
        ),
    ]


    for test in tests:
        if 'error' in test:
            context_manager = pytest.raises(test['error'])
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:
            for sel in test['sel']:
                array_idx = idx._array_index(sel)
                updated_idx = idx._updated_index(array_idx)
                
                assert updated_idx == test['exp_idx'], (updated_idx, test['exp_idx'])
                assert updated_idx.shape == arr[array_idx].shape
                # test short-circuiting if index is unchanged
                if idx == test['exp_idx']:
                    assert id(idx) == id(updated_idx)

def test_Table_construction_and_writing():
    # Can we construct a Table
    np.random.seed(1201)
    tb_vals = np.random.random((5, 3))
    tb = Table(
        data=tb_vals.copy(),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b', 'c', 'd', 'e'),
                ('x', 'y', 'z')
            )
        )
    )
    
    # Catch bad constructions?
    # too few coordinates
    with pytest.raises(ValueError):
        Table(
            data=np.random.random((5, 3)),
            table_index=TableIndex(
                field_names=("dim1", "dim2"),
                field_domains=(
                    ('a', 'b', 'd', 'e'),
                    ('x', 'y', 'z')
                )
            )
        )

    # too many coordinates
    with pytest.raises(ValueError):
        Table(
            data=np.random.random((5, 3)),
            table_index=TableIndex(
                field_names=("dim1", "dim2"),
                field_domains=(
                    ('a', 'b', 'c', 'd', 'e', 'f'),
                    ('x', 'y', 'z')
                )
            )
        )

    # non-unique coordinates
    with pytest.raises(ValueError):
        Table(
            data=np.random.random((5, 3)),
            table_index=TableIndex(
                field_names=("dim1", "dim2"),
                field_domains=(
                    ('a', 'b', 'b', 'd', 'e'),
                    ('x', 'y', 'z')
                )
            )
        )

    # test equality of tables
    data = np.random.random((5, 3))
    tb1 = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b'),
                ('x', 'y', 'z')
            )
        )
    )
    tb2 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]) + 1,
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b'),
                ('x', 'y', 'z')
            )
        )
    )
    tb3 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b'),
                ('x', 'y', 'z')
            )
        )
    )
    tb4 = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('j', 'k'),
                ('x', 'y', 'z')
            )
        )
    )
    assert tb1.equivalent_to(tb2)
    assert tb2.equivalent_to(tb1)
    assert not tb1.equivalent_to(tb3)
    assert not tb1.equivalent_to(tb4)
    assert np.isclose(tb1._data, tb4._data).all()

    # blocking write to underlying array
    with pytest.raises(ValueError):
        tb._data[:] = 1

    # test repr
    tb1_repr = eval(repr(tb1), {**globals(), 'array': np.array})
    assert tb1_repr.equivalent_to(tb1)
    
def test_Table_array_like_interface():
    np.random.seed(1201)
    data = np.random.random((5, 3))
    tb = Table(
        data=data,
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b', 'c', 'd', 'e'),
                ('x', 'y', 'z')
            )
        )
    )
    # Accessing elements in an array-like way
    assert tb["d", "y"] == data[3, 1]

    # Throw error when accessing a non-existent key
    tb["d"]
    with pytest.raises(IndexError):
        tb["d", "w"]

    # Support for some numpy attributes on object
    assert tb.shape == (5, 3)
    assert tb.ndim == 2

    # Castng to a numpy array
    assert isinstance(np.array(tb), np.ndarray)

    
def test_Table_dict_like_interface():
    np.random.seed(1201)
    tb = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],

        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b'),
                ('x', 'y', 'z'),
            ),
        )
    )

    # Can we access outmost dimension as dict keys?
    assert isinstance(tb['a'], Table)
    assert (tb['a']._data == np.array([1, 2, 3])).all()
    assert tb['a', 'x'] == 1

    # Can we slice into subarrays of >2 dims?
    tb_3d = Table(
        data=np.array([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [0, 1, 8],
            ],
        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2", "dim3"),
            field_domains=(
                ('a', 'b'),
                ((6,), (9,)),
                ('x', 'y', 'z'),
            ),
        )
    )
    tb_2d = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        table_index=TableIndex(
            field_names=("dim2", "dim3"),
            field_domains=(
                ((6,), (9,)),
                ('x', 'y', 'z'),
            ),
        )
    )
    assert tb_3d['a'].equivalent_to(tb_2d)

    # Can we iterate over outermost dimension with items()?
    for key, value in tb.items():
        assert tb[key].equivalent_to(value)

def test_ProbabilityTable_and_TableDistribution():
    # Can we access innermost dimension as a Distribution 
    # when representing probabilities?
    tb_probs = ProbabilityTable(
        data=np.array([
            [.5, .25, .25],
            [1/3, 1/3, 1/3],
        ]),
        table_index=TableIndex(
            field_names=("dim1", "dim2"),
            field_domains=(
                ('a', 'b'),
                ('x', 'y', 'z'),
            ),
        )
    )
    assert isinstance(tb_probs['a'], Distribution)
    assert isinstance(tb_probs['a'], Table)
    assert isinstance(tb_probs['a', 'y'], float)

    probs = np.random.random((5, 5, 4, 3))
    probs = probs/probs.sum(axis=(-1, -2), keepdims=True)
    tb_4d = ProbabilityTable(
        data=probs,
        table_index=TableIndex(
            field_names=("dim1", "dim2", "dim3", "dim4"),
            field_domains=(
                tuple("abcde"),
                tuple("fghij"),
                tuple("klmn"),
                tuple("xyz"),
            ),
        )
    )
    tb_4d.probs_start_index = -2
    assert not isinstance(tb_4d, Distribution)
    assert not isinstance(tb_4d['c'], Distribution)
    assert isinstance(tb_4d['c', 'g'], Distribution)
    assert isinstance(tb_4d['c', 'g', 'n'], Distribution)
    assert isinstance(tb_4d['c', 'g', 'n', 'z'], float)
