from itertools import product
from msdm.core.table import AbstractTable

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