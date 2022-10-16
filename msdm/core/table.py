from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from multiprocessing.sharedctypes import Value
import numpy as np
from itertools import product
from typing import Hashable, Mapping, Sequence, Union, Tuple, TypeVar, Any
from msdm.core.distributions.dictdistribution import DictDistribution

from msdm.core.utils.funcutils import cached_property

TableEntry = TypeVar("TableEntry")
IndexValue = TypeVar("IndexValue")

class AbstractTable(ABC):
    # dictionary-like interface
    @abstractmethod
    def __getitem__(self, key: Union[IndexValue,Tuple[IndexValue]]) -> Union[TableEntry,"AbstractTable"]:
        """
        `key` can be a single outermost-variable value of the index,
        or a tuple of variable values, in order of the variables.
        The return value is either a single table entry or another table.
        """
        pass
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def keys(self) -> Sequence[IndexValue]:
        """
        Generates all outer keys in a non-guaranteed order
        """
        pass
    def __iter__(self) -> Sequence[IndexValue]:
        yield from self.keys()
    def items(self) -> Sequence[Tuple[IndexValue,Union[TableEntry,"AbstractTable"]]]:
        yield from ((k, self[k]) for k in self.keys())
    def values(self) -> Sequence[Union[TableEntry,"AbstractTable"]]:
        yield from (self[k] for k in self.keys())
    def get(self, key : IndexValue, default=None) -> Any:
        try:
            return self[key]
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

IndexVariable = namedtuple("IndexVariable", "name values")
class TableIndex:
    """
    An ordered collection of named variables representing
    the indexing scheme of a table.
    """
    def __init__(self, variables : Sequence[IndexVariable]):
        self._variables = variables
    def __getitem__(self, variable_index):
        return self._variables[variable_index]
    @cached_property
    def variable_names(self):
        return tuple([v.name for v in self._variables])
    @cached_property
    def variables(self):
        return tuple(self._variables)
    @cached_property
    def variable_values(self):
        return tuple([v.values for v in self._variables])
    def product(self):
        yield from product(*self.variable_values)
    def product_dicts(self):
        for assignments in self.product():
            yield dict(zip(self.variable_names, assignments))
    def equivalent_to(self, other):
        return (
            self.variable_names == other.variable_names and \
            all([c1 == c2 for c1, c2 in zip(self.variable_values, other.variable_values)])
        )

class Table_repr_html_MixIn(AbstractTable):
    _column_dims_idx = -1 #this is for interpreting a table as a matrix
    def _repr_html_(self):
        import pandas as pd
        pivot = self._column_dims_idx
        variable_names = self.table_index.variable_names
        variable_values = self.table_index.variable_values
        df_data = []
        row_dims = variable_names[:pivot]
        col_dims = variable_names[pivot:]
        row_coords = variable_values[:pivot]
        col_coords = variable_values[pivot:]
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
    A table backed by a numpy array.
    """
    _perform_table_validations = True
    def __init__(
        self,
        data : np.ndarray,
        variable_names : Sequence[Hashable],
        variable_values : Sequence[Tuple[Hashable]],
    ):
        self._data = data
        self._data.setflags(write=False)
        if len(variable_names) != len(variable_values):
            raise ValueError("Different numbers of variable names and values")
        self._index = TableIndex(
            [IndexVariable(n, v) for n, v in zip(variable_names, variable_values)]
        )
        if self._perform_table_validations:
            self._validate_table()

    def _validate_table(self):
        coords_shape = tuple([len(c) for c in self.table_index.variable_values])
        unique_shape = tuple([len(set(c)) for c in self.table_index.variable_values])
        data_shape = self._data.shape
        if not (data_shape == coords_shape == unique_shape):
            raise ValueError(
                f"Total coordinates ({coords_shape}) or unique coordinats ({unique_shape})" + \
                f" do not match dimension lengths ({data_shape})."
            )
        
    # dict-like interface
    def __getitem__(self, table_key):
        if not isinstance(table_key, (tuple, list)):
            table_keys = (table_key,)
        elif table_key in self.table_index.variable_values[0]:
            table_keys = (table_key,)
        else:
            table_keys = table_key
        array_idx = self._array_index(table_keys)
        if len(array_idx) == len(self._data.shape): #array element
            return self._data[array_idx]
        return self._get_subtable(array_idx)
    def _array_index(self, keys):
        try:
            idx = [vals.index(assn) for vals, assn in zip(self.table_index.variable_values, keys)]
        except ValueError:
            raise ValueError(f"{keys} is not in Table")
        return tuple(idx)
    def _get_subtable(self, array_idx):
        return Table(
            data=self._data[array_idx],
            variable_names=self.table_index.variable_names[len(array_idx):],
            variable_values=self.table_index.variable_values[len(array_idx):],
        )
    def keys(self):
        yield from self.table_index.variable_values[0]
    def __len__(self):
        return len(self.table_index.variable_values[0])
    
    # np.array-like interface
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
            f"variable_names={repr(self.table_index.variable_names)},\n" + \
            f"variable_values={repr(self.table_index.variable_values)})"

class ProbabilityTable(Table):
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
                variable_names=self.table_index.variable_names[len(array_idx):],
                variable_values=self.table_index.variable_values[len(array_idx):],
            )
        return Table._get_subtable(self, array_idx)

class TableDistribution(Table,DictDistribution):
    """DictDistribution backed by a numpy array."""
    pass