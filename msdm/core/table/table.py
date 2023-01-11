from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from typing import Sequence, Union, Tuple, TypeVar, Any
from msdm.core.distributions.dictdistribution import DictDistribution
from msdm.core.table.tableindex import FieldValue, TableIndex

TableKey = Union[FieldValue,Tuple[FieldValue,...]]
TableEntry = TypeVar("TableEntry")
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

from msdm.core.table.tablemisc import Table_repr_html_MixIn
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

    def __getitem__(self, selector):
        array_index = self.table_index._array_index(selector)
        new_table_index = self.table_index._updated_index(array_index)
        if new_table_index == self.table_index:
            return self
        new_data = self._data[array_index]
        if isinstance(new_data, np.ndarray):
            return self.__class__(
                data=new_data,
                table_index=new_table_index
            )
        return new_data
        
    def keys(self):
        yield from self.table_index.fields[0].domain

    def __len__(self):
        return len(self.table_index.fields[0].domain)

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
            self.table_index == other.table_index and \
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
    def __getitem__(self, selector):
        array_index = self.table_index._array_index(selector)
        new_table_index = self.table_index._updated_index(array_index)
        if new_table_index == self.table_index:
            return self
        new_data = self._data[array_index]
        if isinstance(new_data, np.ndarray):
            if new_data.ndim <= (-self.probs_start_index): 
                return TableDistribution(
                    data=new_data,
                    table_index=new_table_index
                )
            else:
                return self.__class__(
                    data=new_data,
                    table_index=new_table_index
                )
        return new_data

class TableDistribution(Table,DictDistribution):
    """DictDistribution backed by a numpy array."""
    pass