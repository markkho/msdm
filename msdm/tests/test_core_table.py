import numpy as np
import string
from itertools import product
from msdm.core.table import Table, ProbabilityTable
from msdm.core.tableindex import TableIndex, Field
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

def test_TableIndex_compatibility():
    fields = [
        Field('a', (0, 1)), 
        Field('b', ("x", "y", "z"))
    ]
    idx = TableIndex(fields=fields)

    comp_fields = [
        Field('b', ("x", "z", "y")),
        Field('a', (1, 0)), 
    ]
    comp_idx = TableIndex(fields=comp_fields)

    incomp_fields1 = [
        Field('c', (0, 1)), 
        Field('b', ("x", "y", "z"))
    ]
    incomp_idx1 = TableIndex(fields=incomp_fields1)

    incomp_fields2 = [
        Field('a', (0, 1)), 
        Field('b', ("w", "y", "z"))
    ]
    incomp_idx2 = TableIndex(fields=incomp_fields2)

    incomp_fields3 = [
        Field('a', (0, 1, 2)), 
        Field('b', ("x", "y", "z"))
    ]
    incomp_idx3 = TableIndex(fields=incomp_fields3)

    assert idx.compatible_with(comp_idx)
    assert not idx.compatible_with(incomp_idx1)
    assert not idx.compatible_with(incomp_idx2)
    assert not idx.compatible_with(incomp_idx3)

def test_TableIndex_reindexing_small():
    fields = [
        Field('a', (0, 1)), 
        Field('b', ("x", "y", "z"))
    ]
    idx = TableIndex(fields=fields)
    
    new_fields = [
        Field('b', ("y", "z", "x")),
        Field('a', (0, 1)), 
    ]
    new_idx = TableIndex(fields=new_fields)
    
    field_permutations, domain_permutations = idx.reindexing_permutations(new_idx)
    assert field_permutations == (1, 0)
    assert domain_permutations == ((0, 1), (1, 2, 0))

def test_random_TableIndex_reindexing():
    # Generate a bunch of random table indexes and random permutations
    for _ in range(100):
        random_TableIndex_reindexing_test(np.random.randint(1, int(1e9)))

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
    try:
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
        assert False
    except ValueError:
        pass

    # too many coordinates
    try:
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
        assert False
    except ValueError:
        pass

    # non-unique coordinates
    try:
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
    try:
        tb._data[:] = 1
        assert False
    except ValueError:
        pass
    # test repr
    tb1_repr = eval(repr(tb1), {**globals(), 'array': np.array})
    assert tb1_repr.equivalent_to(tb1)
    
def test_random_Table_reindexing():
    # Generate a bunch of random tables and random reindexings of them
    for _ in range(100):
        random_Table_reindexing_test(np.random.randint(1, int(1e9)))

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
    assert not isinstance(tb_4d['c', 'g', 'n', 'z'], Distribution)

################################
#        Helper Functions      #
################################
def random_TableIndex_reindexing_test(seed):
    """
    Generate a random TableIndex and random permutation of it to make sure
    that the automatic reindexing is correct.
    """
    np.random.seed(seed)
    table_index = generate_random_TableIndex(
        names=tuple("abcxyz01234569") + ((), (1, ), (2, 3, 4), ("123",), frozenset([])),
        values=list(range(100))+list(string.ascii_letters)+list(product("abcdefg", ((), (1,), (2, 3)))),
        max_ndim=20,
        max_domain_size=10,
        seed=np.random.randint(1, int(1e9))
    )
    new_table_index, field_permutation, domain_permutations = \
        random_TableIndex_permutation(
            table_index=table_index,
            seed=np.random.randint(1, int(1e9))
    )
    comp_field_perm, comp_domain_perms = table_index.reindexing_permutations(new_table_index)
    try:
        assert comp_field_perm == field_permutation
        for comp_domain_perm, domain_perm in zip(comp_domain_perms, domain_permutations):
            assert comp_domain_perm == domain_perm, (comp_domain_perm, domain_perm)
    except AssertionError:
        raise AssertionError(f"Reindexing failed for seed {seed}")
        
def random_Table_reindexing_test(seed):
    # Randomly construct a table index and table values
    # randomly permute the index and reindex the table
    # test that the data in the new table is reindexed properly
    np.random.seed(seed)
    table_index = generate_random_TableIndex(
        names=tuple("abcxyz01234569") + ((), (1, ), (2, 3, 4), ("123",), frozenset([])),
        values=list(range(100))+list(string.ascii_letters)+list(product("abcdefg", ((), (1,), (2, 3)))),
        max_ndim=10,
        max_domain_size=6,
        seed=np.random.randint(1, int(1e9))
    )
    tb = Table(
        data=np.random.random(table_index.shape),
        table_index=table_index
    )
    new_table_index, field_permutation, domain_permutations = \
        random_TableIndex_permutation(
            table_index=table_index,
            seed=np.random.randint(1, int(1e9))
    )
    new_tb = tb.reindex(new_index=new_table_index)

    # Iterate through the source indexes and destination indexes explicitly
    # to build up the new table.
    # Careful - this is a bit tricky!
    src_data = np.array(tb)
    dest_data = np.zeros_like(src_data)
    dest_arr_idx = [range(len(f.domain)) for f in tb.table_index.fields]
    dest_arr_idx = list(product(*dest_arr_idx))
    src_arr_idx = list(product(*domain_permutations))
    assert len(dest_arr_idx) == len(src_arr_idx)
    for src_i, dest_i in zip(src_arr_idx, dest_arr_idx):
        dest_data[dest_i] = src_data[src_i]
    dest_data = dest_data.transpose(*field_permutation)
    
    try:
        assert (dest_data == np.array(new_tb)).all()
    except AssertionError:
        raise AssertionError(f"Reindexing failed for seed {seed}")
    
def generate_random_TableIndex(names, values, max_ndim=1000, max_domain_size=1000, seed=None):
    np.random.seed(seed)
    ndim = np.random.randint(1, min(len(names), max_ndim))
    fields = []
    names = [n for n in names]
    np.random.shuffle(names)
    for dim in range(ndim):
        np.random.shuffle(values)
        domain_size = np.random.randint(1, min(len(values), max_domain_size))
        fields.append(
            Field(
                name=names.pop(),
                domain=tuple(values[:domain_size])
            )
        )
    return TableIndex(fields=fields)

def random_TableIndex_permutation(table_index : TableIndex, seed=None):
    np.random.seed(seed)
    domain_permutations = [list(range(len(domain))) for domain in table_index.field_domains]
    for order in domain_permutations:
        np.random.shuffle(order)
    field_permutation = list(range(len(table_index)))
    np.random.shuffle(field_permutation)
    new_fields = []
    for field, domain_permutation in zip(table_index.fields, domain_permutations):
        new_fields.append(
            Field(
                name=field.name,
                domain=tuple([field.domain[i] for i in domain_permutation])
            )
        )
    new_fields = [new_fields[i] for i in field_permutation]
    new_table_index = TableIndex(fields=new_fields)
    field_permutation = tuple(field_permutation)
    domain_permutations = [tuple(p) for p in domain_permutations]
    return new_table_index, field_permutation, domain_permutations
