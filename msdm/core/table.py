from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple
import numpy as np
from itertools import product
from typing import Hashable, Sequence, Union, Tuple, TypeVar, Any
from msdm.core.distributions.dictdistribution import DictDistribution

from msdm.core.utils.funcutils import cached_property

TableEntry = TypeVar("TableEntry")
FieldValue = TypeVar("FieldValue")
TableKey = Union[FieldValue,Tuple[FieldValue,...]]
TableValue = Union[TableEntry,"AbstractTable"]

class AbstractTable(ABC):
    """
    This defines the generic interface for a Table, which 
    essentially consists of two components: 

      1. A TableIndex, which is a sequence of IndexField's, 
      each defined by a name and domain.

      2. An underlying data structure to index into. An  
      assignment to all IndexFields's maps to a single
      primitive data element in this structure.
    

    The motivation for Tables is to enable
    interface elements from several different Python 
    collection objects for efficiently building, accessing, and 
    manipulating arbitrary indexed data:
    
      1. Tables behave like dictionaries or nested dictionaries,
      by enabling `__getitem__` and `get()` access with values of the
      outermost IndexField's or an ordered tuple of 
      IndexField's assignments that start from the outside in. 
      This will return either a sub-Table or primitive data element.

      2. Iteration over the *outermost* IndexField and/or
      corresponding sub-Tables with `items()`, `values()`, 
      `keys()`, and `__iter__()` that mimics standard `dict`s.

      3. Numpy-array-like access, composition, and 
      manipulation of the underlying data that preserves
      index-data mappings (currently limited).

      4. Pandas-like usability in notebooks. 
    """

    # dictionary-like interface
    @abstractmethod
    def __getitem__(self, key: TableKey) -> TableValue:
        """
        `key` can be a single outermost-field's assignment,
        or a tuple of field values, in order of the fields.
        The return value is either a single table entry or another table.
        """
        pass
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def keys(self) -> Sequence[FieldValue]:
        """
        Generates all outer keys in a non-guaranteed order
        """
        pass
    def __iter__(self) -> Sequence[FieldValue]:
        yield from self.keys()
    def items(self) -> Sequence[Tuple[FieldValue,TableValue]]:
        yield from ((k, self[k]) for k in self.keys())
    def values(self) -> Sequence[TableValue]:
        yield from (self[k] for k in self.keys())
    def get(self, key : TableKey, default=None) -> Any:
        try:
            value : TableValue =  self[key]
            return value
        except KeyError:
            return default
    
    # np.array-like interface (currently limited)
    @abstractmethod
    def __array__(self, dtype=None):
        pass
    def __array_ufunc__(self, ufunc, method, *args, **kws):
        raise NotImplementedError
    def __getattr__(self, attr):
        if attr in ("shape", "ndim"):
            return getattr(self.__array__(), attr)
        else:
            raise AttributeError(f"'{self.__class__}' object has no attribute '{attr}'")    
    
    # AbstractTable interface
    @abstractproperty
    def table_index(self) -> "TableIndex": pass

class IndexField(NamedTuple):
    name : Hashable
    domain : Sequence[FieldValue]
    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, domain={repr(self.domain)})"

class TableIndex:
    """
    An ordered collection of named fields representing
    the indexing scheme of a table.
    """
    def __init__(
        self,
        *,
        field_names : Sequence[Hashable] = None,
        field_domains : Sequence[Tuple[Hashable,...]] = None,
        fields : Sequence[IndexField] = None,
    ):
        if fields is None:
            if len(field_names) != len(field_domains):
                raise ValueError("Different numbers of fields names and domains")
            fields = [IndexField(n, v) for n, v in zip(field_names, field_domains)]
        self._fields = fields
    def __getitem__(self, field_selection):
        if isinstance(field_selection, slice):
            return self.__class__(fields=self._fields[field_selection])
        return self._fields[field_selection]
    @cached_property
    def field_names(self):
        return tuple([v.name for v in self._fields])
    @cached_property
    def fields(self):
        return tuple(self._fields)
    @cached_property
    def field_domains(self):
        return tuple([v.domain for v in self._fields])
    def product(self):
        yield from product(*self.field_domains)
    def product_dicts(self):
        for assignments in self.product():
            yield dict(zip(self.field_names, assignments))
    def equivalent_to(self, other : "TableIndex"):
        return (
            self.field_names == other.field_names and \
            all([c1 == c2 for c1, c2 in zip(self.field_domains, other.field_domains)])
        )
    def __repr__(self):
        return f"{self.__class__.__name__}(fields={repr(self._fields)})"


class Table_repr_html_MixIn(AbstractTable):
    _column_dims_idx = -1 #this is for interpreting a table as a matrix
    def _repr_html_(self):
        import pandas as pd
        pivot = self._column_dims_idx
        field_names = self.table_index.field_names
        field_domains = self.table_index.field_domains
        df_data = []
        row_dims = field_names[:pivot]
        col_dims = field_names[pivot:]
        row_coords = field_domains[:pivot]
        col_coords = field_domains[pivot:]
        df_index = []
        df_cols = list(product(*col_coords))
        df_index = list(product(*row_coords))
        for row_idx in product(*row_coords):
            row = []
            for col_idx in product(*col_coords):
                val = self[row_idx + col_idx]
                row.append(val)
            df_data.append(row)
        if len(sum(df_cols, ())) > 0:
            df_cols = pd.MultiIndex.from_tuples(df_cols, names=col_dims)
        else:
            df_cols = ("values",)
        if len(sum(df_index, ())) > 0:
            df_index = pd.MultiIndex.from_tuples(df_index, names=row_dims)
        else:
            df_index = ("values",)
        html_table = pd.DataFrame(
            df_data,
            columns=df_cols,
            index=df_index
        ).to_html()
        return html_table

class Table(Table_repr_html_MixIn,AbstractTable):
    """
    A Table backed by a numpy array.
    """
    _perform_table_validation = True
    def __init__(
        self,
        data : np.ndarray,
        table_index : TableIndex,
    ):
        self._data = data
        self._data.setflags(write=False)
        self._index = table_index
        if self._perform_table_validation:
            self._validate_table()

    def _validate_table(self):
        coords_shape = tuple([len(c) for c in self.table_index.field_domains])
        unique_shape = tuple([len(set(c)) for c in self.table_index.field_domains])
        data_shape = self._data.shape
        if not (data_shape == coords_shape == unique_shape):
            raise ValueError(
                f"Total coordinates ({coords_shape}) or unique coordinats ({unique_shape})" + \
                f" do not match dimension lengths ({data_shape})."
            )

    def __getitem__(self, table_key):
        if not isinstance(table_key, (tuple, list)):
            table_keys = (table_key,)
        elif table_key in self.table_index.field_domains[0]:
            table_keys = (table_key,)
        else:
            table_keys = table_key
        array_idx = self._array_index(table_keys)
        if len(array_idx) == len(self._data.shape): #array element
            return self._data[array_idx]
        return self._get_subtable(array_idx)

    def _array_index(self, keys):
        try:
            idx = [vals.index(assn) for vals, assn in zip(self.table_index.field_domains, keys)]
        except ValueError:
            raise ValueError(f"{keys} is not in Table")
        return tuple(idx)

    def _get_subtable(self, array_idx):
        return Table(
            data=self._data[array_idx],
            table_index=self.table_index[len(array_idx):]
        )

    def keys(self):
        yield from self.table_index.field_domains[0]

    def __len__(self):
        return len(self.table_index.field_domains[0])

    def __array__(self, dtype=None):
        return self._data

    @property
    def table_index(self):
        return self._index
    
    def equivalent_to(
        self, other: "Table", *,
        # These tolerances are copied from `np.isclose()`
        rtol=1e-05, atol=1e-08,
    ) -> bool:
        return (
            self.shape == other.shape and \
            self.table_index.equivalent_to(other.table_index) and \
            np.isclose(self._data, other._data, atol=atol, rtol=rtol).all()
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" +\
            f"data={repr(self._data)},\n" + \
            f"table_index={repr(self.table_index)})"

class ProbabilityTable(Table):
    """
    Table that represents conditional probability distributions.
    """

    #the starting dimension over which entries are a single probability distribution
    probs_start_index = -1 
    def _get_subtable(self, array_idx):
        if self.probs_start_index < 0:
            probs_start_index = len(self._data.shape) + self.probs_start_index
        else:
            probs_start_index = self.probs_start_index
        if len(array_idx) >= probs_start_index:
            return TableDistribution(
                data=self._data[array_idx],
                table_index=self.table_index[len(array_idx):]
            )
        return Table._get_subtable(self, array_idx)

class TableDistribution(Table,DictDistribution):
    """DictDistribution backed by a numpy array."""
    pass