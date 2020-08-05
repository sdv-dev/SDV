"""Miscellaneous utility functions."""

from IPython.core.display import HTML


def display_tables(tables, max_rows=10, datetime_fmt='%Y-%m-%d %H:%M:%S'):
    """Display mutiple tables side by side on a Jupyter Notebook.

    Args:
        tables (dict[str, DataFrame]):
            ``dict`` containing table names and pandas DataFrames.
        max_rows (int):
            Max rows to show per table. Defaults to 10.
        datetime_fmt (str):
            Format with which to display datetime columns.
    """
    names = []
    data = []
    for name, table in tables.items():
        table = table.copy()
        for column in table.columns:
            column_data = table[column]
            if column_data.dtype.kind == 'M':
                table[column] = column_data.dt.strftime(datetime_fmt)

        names.append('<td><b>{}</b></td>'.format(name))
        data.append('<td>{}</td>'.format(table.head(max_rows)._repr_html_()))

    return HTML('<table><tr>{}</tr><tr>{}</tr></table>'.format(
        ''.join(names),
        ''.join(data),
    ))
