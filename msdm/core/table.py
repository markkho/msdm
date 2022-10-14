import numpy as np
from itertools import product
from typing import Hashable, Mapping, Sequence, Union, Tuple
from msdm.core.distributions.dictdistribution import DictDistribution

class Table(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A table maps an unordered set of multi-dimensional 
    keys to real values. It can be thought of as a
    np.array with labeled dimensions and indices.
    """
    def __init__(
        self,
        data : np.array,
        dims : Sequence[str],
        coords : Union[Tuple[Sequence[Hashable]], Mapping[str, Sequence[Hashable]]],
    ):
        self._data = data
        self._data.setflags(write=False)
        self._dims = dims
        self._coords = coords
        self._column_dims_idx = -1
    
    def _validate_table(self):
        assert isinstance(self._coords, (tuple, list))
        assert isinstance(self._dims, (tuple, list))
        assert len(self._coords) == len(self._dims), "Number of coordinates and dimensions don't match"
        coords_shape = tuple([len(c) for c in self._coords])
        unique_shape = tuple([len(set(c)) for c in self._coords])
        data_shape = self._data.shape
        if not (data_shape == coords_shape == unique_shape):
            raise ValueError(
                f"Total coordinates ({coords_shape}) or unique coordinats ({unique_shape})" + \
                f" do not match dimension lengths ({data_shape})."
            )
        
    # dict-like interface
    def __getitem__(self, key):
        if not isinstance(key, (tuple, list)):
            keys = (key,)
        elif key in self._coords[0]:
            keys = (key,)
        else:
            keys = key
        idx = self._key_indices(keys)
        if len(idx) == len(self._data.shape): #array element
            return self._data[idx]
        return self._get_subtable(idx)
    def _key_indices(self, keys):
        idx = tuple([coords.index(k) for coords, k in zip(self._coords, keys)])
        return idx
    def _get_subtable(self, coords_idx):
        return Table(
            data=self._data[coords_idx],
            dims=self._dims[len(coords_idx):],
            coords=self._coords[len(coords_idx):],
        )
    def items(self):
        yield from ((k, self[k]) for k in self.keys())
    def keys(self):
        yield from self._coords[0]
    def values(self):
        yield from (self[k] for k in self.keys())
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    def __len__(self):
        return len(self._coords[0])
    def __iter__(self):
        yield from self.keys()
    def isclose(
        self, other: "Table", *,
        # These tolerances are copied from `np.isclose()`
        rtol=1e-05, atol=1e-08,
    ) -> bool:
        return (
            self.shape == other.shape and \
            self._dims == other._dims and \
            all([c1 == c2 for c1, c2 in zip(self._coords, other._coords)]) and \
            np.isclose(self._data, other._data, atol=atol, rtol=rtol).all()
        )
    
    # indirect support for np.array-like interface
    def __getattr__(self, attr):
        if attr in ("shape", "ndim"):
            return getattr(self._data, attr)
        else:
            raise AttributeError(f"'{self.__class__}' object has no attribute '{attr}'")    
    def __array__(self, dtype=None):
        return self._data
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" +\
            f"data={repr(self._data)},\n" + \
            f"coords={repr(self._coords)},\n" + \
            f"dims={repr(self._dims)})"
    
    def _repr_html_(self):
        import pandas as pd
        pivot = self._column_dims_idx
        df_data = []
        row_dims = self._dims[:pivot]
        col_dims = self._dims[pivot:]
        row_coords = self._coords[:pivot]
        col_coords = self._coords[pivot:]
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

class ProbabilityTable(Table):
    # Probability tables represent and return distributions 
    # if indexed into a column dimension but otherwise behave like tables
    def _get_subtable(self, coords_idx):
        assert self._column_dims_idx in (-1, self.ndim - 1), \
            f"Currently, for {self.__class__.__name__}, only the last dimension can be a column dimension " +\
            f" but column dimensions start at {self._column_dims_idx}"
        result_dimensionality = len(self._data.shape) - len(coords_idx)
        if result_dimensionality == 1:
            return TableDistribution(
                data=self._data[coords_idx],
                dims=self._dims[len(coords_idx):],
                coords=self._coords[len(coords_idx):]
            )
        return Table._get_subtable(self, coords_idx)

class TableDistribution(Table,DictDistribution):
    """DictDistribution backed by a numpy array."""
    pass