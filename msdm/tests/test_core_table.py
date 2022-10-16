import numpy as np
from msdm.core.table import Table, ProbabilityTable, IndexVariable, TableIndex
from msdm.core.distributions import Distribution 

def test_TableIndex():
    variables = [
        IndexVariable('a', (0, 1)), 
        IndexVariable('b', ("x", "y", "z"))
    ]
    idx = TableIndex(variables)
    assert idx.variable_names == ('a', 'b')
    for vi, v in enumerate(idx.variables):
        assert v.name == variables[vi].name
        assert v.values == variables[vi].values
    assert idx.variable_values == ((0, 1), ("x", "y", "z"))
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

def test_Table_construction_and_writing():
    # Can we construct a Table
    np.random.seed(1201)
    tb_vals = np.random.random((5, 3))
    tb = Table(
        data=tb_vals.copy(),
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b', 'c', 'd', 'e'),
            ('x', 'y', 'z')
        )
    )
    
    # Catch bad constructions?
    # too few coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            variable_names=("dim1", "dim2"),
            variable_values=(
                ('a', 'b', 'd', 'e'),
                ('x', 'y', 'z')
            )
        )
        assert False
    except ValueError:
        pass

    # too many coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            variable_names=("dim1", "dim2"),
            variable_values=(
                ('a', 'b', 'c', 'd', 'e', 'f'),
                ('x', 'y', 'z')
            )
        )
        assert False
    except ValueError:
        pass

    # non-unique coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            variable_names=("dim1", "dim2"),
            variable_values=(
                ('a', 'b', 'b', 'd', 'e'),
                ('x', 'y', 'z')
            )
        )
        assert False
    except ValueError:
        pass
    # test equality of tables
    data = np.random.random((5, 3))
    tb1 = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b'),
            ('x', 'y', 'z')
        )
    )
    tb2 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]) + 1,
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b'),
            ('x', 'y', 'z')
        )
    )
    tb3 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]),
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b'),
            ('x', 'y', 'z')
        )
    )
    tb4 = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('j', 'k'),
            ('x', 'y', 'z')
        )
    )
    assert tb1.equivalent_to(tb2)
    assert tb2.equivalent_to(tb1)
    assert not tb1.equivalent_to(tb3)
    assert not tb1.equivalent_to(tb4)
    assert np.isclose(tb1._data, tb4._data).all()

    # blocking write to underlying array
    try:
        tb._data[:] = 1
        assert False
    except ValueError:
        pass
    # test repr
    tb1_repr = eval(repr(tb1), {**globals(), 'array': np.array})
    assert tb1_repr.equivalent_to(tb1)
    
def test_Table_array_like_interface():
    np.random.seed(1201)
    data = np.random.random((5, 3))
    tb = Table(
        data=data,
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b', 'c', 'd', 'e'),
            ('x', 'y', 'z')
        )
    )
    # Accessing elements in an array-like way
    assert tb["d", "y"] == data[3, 1]

    # Throw error when accessing a non-existent key
    tb["d"]
    try:
        tb["d", "w"]
        assert False
    except ValueError:
        pass

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
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b'),
            ('x', 'y', 'z'),
        ),
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
        variable_names=("dim1", "dim2", "dim3"),
        variable_values=(
            ('a', 'b'),
            ((6,), (9,)),
            ('x', 'y', 'z'),
        ),
    )
    tb_2d = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        variable_names=("dim2", "dim3"),
        variable_values=(
            ((6,), (9,)),
            ('x', 'y', 'z'),
        ),
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
        variable_names=("dim1", "dim2"),
        variable_values=(
            ('a', 'b'),
            ('x', 'y', 'z'),
        ),
    )
    assert isinstance(tb_probs['a'], Distribution)
    assert isinstance(tb_probs['a'], Table)

    probs = np.random.random((5, 5, 4, 3))
    probs = probs/probs.sum(axis=(-1, -2), keepdims=True)
    tb_4d = ProbabilityTable(
        data=probs,
        variable_names=("dim1", "dim2", "dim3", "dim4"),
        variable_values=(
            tuple("abcde"),
            tuple("fghij"),
            tuple("klmn"),
            tuple("xyz"),
        ),
    )
    tb_4d.probs_start_index = -2
    assert not isinstance(tb_4d, Distribution)
    assert not isinstance(tb_4d['c'], Distribution)
    assert isinstance(tb_4d['c', 'g'], Distribution)
    assert isinstance(tb_4d['c', 'g', 'n'], Distribution)
    assert not isinstance(tb_4d['c', 'g', 'n', 'z'], Distribution)