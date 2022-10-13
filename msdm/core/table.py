import numpy as np
from itertools import product
from typing import Hashable, Sequence
from msdm.core.utils.funcutils import method_cache
from msdm.core.distributions import DictDistribution

class Table(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self,
        values : np.array,
        # dims : Sequence[str],
        coords : tuple[Sequence[Hashable]],
        _probabilities=False,
        _coords_indices=None
    ):
        self._values = values
        self._values.setflags(write=False)
        self._coords = tuple(coords)
        if _coords_indices is None:
            _coords_indices = [{i: ii for ii, i in enumerate(dim)} for dim in coords]
        assert values.shape == tuple([len(dim) for dim in _coords_indices])
        self._coords_indices = _coords_indices
        self._probabilities = _probabilities
        
    # dict-like interface
    def __getitem__(self, key):
        if not isinstance(key, (tuple, list)):
            keys = (key,)
        elif key in self._coords_indices[0]:
            keys = (key,)
        else:
            keys = key
        idx = tuple([dim[k] for dim, k in zip(self._coords_indices, keys)])
        if len(idx) == len(self._values.shape): #array element
            return self._values[idx]
        return self._get_subspace(idx)
    
    @method_cache
    def _get_subspace(self, idx):
        result_dimensionality = len(self._values.shape) - len(idx)
        if self._probabilities and result_dimensionality == 1:
            # by convention, probabilities are over the last dimension
            # if its probabilities, return a Distribution so we can sample, etc.
            probs = self._values[idx]
            return DictDistribution(zip(self._coords[len(idx)], probs))
        # otherwise, return a TabularMap of that subspace
        return Table(
            values=self._values[idx],
            coords=self._coords[len(idx):],
            _probabilities=self._probabilities,
            _coords_indices=self._coords_indices[len(idx):]
        )
    def items(self):
        yield from ((k, self[k]) for k in self.keys())
    # def keys(self):
    #     yield from self._dimension_indices[0]
    # def values(self):
    #     yield from (self[k] for k in self.keys())
    def __len__(self):
        return len(self._coords[0])
    # def __iter__(self):
    #     yield from self.keys()
    
    # np.array-like interface
    def __getattr__(self, attr):
        return getattr(self._values, attr)
    def __array__(self, dtype=None):
        return self._values
    @property
    def __array_interface__(self):
        return self._values.__array_interface__
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" +\
            f"values={repr(self._values)},\n" + \
            f"dimensions={repr(self._coords)},\n" + \
            f"probabilities={self._probabilities},\n" + \
            f"dimension_indices={self._coords_indices})"
    
    def _repr_html_(self):
        rowdims = len(self._coords) - 1
        colnames = \
            ['<th></th>']*rowdims + \
            [f'<th>{k}</th>' for k in self._coords[-1]]
        if rowdims == 0:
            # its a single vector
            colnames = []
            rows = []
            for k, v in self.items():
                rows.append(f"<tr><th>{k}</th><td>{v:.2f}</td>")
        elif rowdims == 1:
            # its 2d
            rows = []
            rownames = self._coords[0]
            for rowname in rownames:
                rowdata = self[rowname]
                rows.extend([
                    "<tr>",
                    f"<th>{rowname}</th>",
                    *[f"<td>{d:.2f}</td>" for d in rowdata.values()],
                    "</tr>"
                ])
        else:
            # its (n > 2)d
            rows = []
            rownames = list(product(*self._coords[:-1]))
            for rowname in rownames:
                rowdata = self[rowname]
                rows.extend([
                    "<tr>",
                    *[f"<th>{k}</th>" for k in rowname],
                    *[f"<td>{d:.2f}</td>" for d in rowdata.values()],
                    "</tr>"
                ])
        html_table = ''.join([
            '<table border="1" class="dataframe">',
            '<thead>',
            '<tr style="text-align: right;">',
            *colnames,
            '</tr>',
            '</thead>',
            '<tbody>',
            *rows,
            "</tbody>",
            "</table>"
        ])
        return html_table