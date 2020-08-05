from IPython.core.display import HTML


def display_tables(tables, max_rows=10, datetime_fmt='%Y-%m-%d %H:%M:%S'):
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
