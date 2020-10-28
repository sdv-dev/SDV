"""Miscellaneous utility functions."""


def display_tables(tables, max_rows=10, datetime_fmt='%Y-%m-%d %H:%M:%S', row=True):
    """Display mutiple tables side by side on a Jupyter Notebook.

    Args:
        tables (dict[str, DataFrame]):
            ``dict`` containing table names and pandas DataFrames.
        max_rows (int):
            Max rows to show per table. Defaults to 10.
        datetime_fmt (str):
            Format with which to display datetime columns.
    """
    # Import here to avoid making IPython a hard dependency
    from IPython.core.display import HTML

    names = []
    data = []
    for name, table in tables.items():
        table = table.copy()
        for column in table.columns:
            column_data = table[column]
            if column_data.dtype.kind == 'M':
                table[column] = column_data.dt.strftime(datetime_fmt)

        names.append('<td style="text-align:left"><b>{}</b></td>'.format(name))
        data.append('<td>{}</td>'.format(table.head(max_rows).to_html(index=False)))

    if row:
        html = '<table><tr>{}</tr><tr>{}</tr></table>'.format(
            ''.join(names),
            ''.join(data),
        )
    else:
        rows = [
            '<tr>{}</tr><tr>{}</tr>'.format(name, table)
            for name, table in zip(names, data)
        ]
        html = '<table>{}</table>'.format(''.join(rows))

    return HTML(html)
