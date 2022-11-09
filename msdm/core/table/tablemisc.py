from itertools import product
from msdm.core.table.table import AbstractTable

class Table_repr_html_MixIn(AbstractTable):
    _column_dims_idx = -1 #this is for interpreting a table as a matrix
    def _repr_html_(self, df_kws=None):
        if df_kws is None:
            df_kws = dict(float_format=lambda f: f"{f:.2f}")
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
            index=df_index,
        ).to_html(**df_kws)
        return html_table

class dataclass_repr_html_MixIn:
    def _repr_html_(self, float_format=lambda f: f"{f:.2f}"):
        import dataclasses
        html_res = [f"<h2>{self.__class__.__name__}</h2>"]
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            try:
                html_res.append(f"<div><h3>{field.name}: </h3>{value._repr_html_()}</div>")
            except AttributeError:
                if isinstance(value,float):
                    html_res.append(f"<div><h3>{field.name}: </h3>{float_format(value)}</div>")
                else:
                    html_res.append(f"<div><h3>{field.name}: </h3>{repr(value)}</div>")
            html_res.append('<hr>')

        return ''.join(html_res)
