import numpy as np
from msdm.core.table import Table, ProbabilityTable
from msdm.core.distributions import Distribution

def test_Table_construction_and_writing():
    # Can we construct a Table and catch bad constructions?
    np.random.seed(1201)
    tb_vals = np.random.random((5, 3))
    tb = Table(
        data=tb_vals.copy(),
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b', 'c', 'd', 'e'],
            ['x', 'y', 'z']
        )
    )
    
    # dictionary coords constructor
    tb_w_dict = Table(
        data=tb_vals.copy(),
        dims=("dim1", "dim2"),
        coords={
            'dim1': ['a', 'b', 'c', 'd', 'e'],
            'dim2': ['x', 'y', 'z']
        }
    )
    assert tb.isclose(tb_w_dict)

    # too few coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            dims=("dim1", "dim2"),
            coords=(
                ['a', 'b', 'd', 'e'],
                ['x', 'y', 'z']
            )
        )
        assert False
    except ValueError:
        pass

    # too many coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            dims=("dim1", "dim2"),
            coords=(
                ['a', 'b', 'c', 'd', 'e', 'f'],
                ['x', 'y', 'z']
            )
        )
        assert False
    except ValueError:
        pass

    # non-unique coordinates
    try:
        Table(
            data=np.random.random((5, 3)),
            dims=("dim1", "dim2"),
            coords=(
                ['a', 'b', 'b', 'd', 'e'],
                ['x', 'y', 'z']
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
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b'],
            ['x', 'y', 'z']
        )
    )
    tb2 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]) + 1,
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b'],
            ['x', 'y', 'z']
        )
    )
    tb3 = Table(
        data=np.array([
            [0, 1, 2],
            [3, 4, 5],
        ]),
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b'],
            ['x', 'y', 'z']
        )
    )
    tb4 = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        dims=("dim1", "dim2"),
        coords=(
            ['j', 'k'],
            ['x', 'y', 'z']
        )
    )
    assert tb1.isclose(tb2)
    assert tb2.isclose(tb1)
    assert not tb1.isclose(tb3)
    assert not tb1.isclose(tb4)
    assert np.isclose(tb1._data, tb4._data).all()

    # blocking write to underlying array
    try:
        tb._data[:] = 1
        assert False
    except ValueError:
        pass
    # test repr
    tb1_repr = eval(repr(tb1), {**globals(), 'array': np.array})
    assert tb1_repr.isclose(tb1)
    
def test_Table_array_like_interface():
    np.random.seed(1201)
    tb = Table(
        data=np.random.random((5, 3)),
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b', 'c', 'd', 'e'],
            ['x', 'y', 'z']
        )
    )
    # Support for some numpy attributes on object
    assert tb.shape == (5, 3)
    assert tb.ndim == 2

    # Can we run numpy functions via __array__() and get back an np.array?
    assert np.mean(tb) == tb._data.mean()
    assert isinstance(np.log(tb), np.ndarray)
    assert isinstance(tb * 2, np.ndarray)
    
def test_Table_dict_like_interface():
    np.random.seed(1201)
    tb = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],

        ]),
        dims=("dim1", "dim2"),
        coords=(
            ['a', 'b'],
            ['x', 'y', 'z'],
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
        dims=("dim1", "dim2", "dim3"),
        coords=(
            ['a', 'b'],
            [(6,), (9,)],
            ['x', 'y', 'z'],
        ),
    )
    tb_2d = Table(
        data=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        dims=("dim2", "dim3"),
        coords=(
            [(6,), (9,)],
            ['x', 'y', 'z'],
        ),
    )
    assert tb_3d['a'].isclose(tb_2d)

    # Can we iterate over outermost dimension with items()?
    for key, value in tb.items():
        assert tb[key].isclose(value)

# def test_ProbabilityTable_and_TableDistribution():