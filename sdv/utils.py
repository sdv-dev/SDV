"""Miscellaneous utility functions."""
import pkg_resources


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


def get_package_versions(model=None):
    """Get the package versions for SDV libraries.

    Args:
        model (object or None):
            If model is not None, also store the SDV library versions relevant to this model.
    Returns:
        dict:
            A mapping of library to current version.
    """
    versions = {
        'sdv': pkg_resources.get_distribution('sdv').version,
        'rdt': pkg_resources.get_distribution('rdt').version,
    }
    if model is not None:
        lib = model.__class__.__module__.split('.')[0]
        versions[lib] = pkg_resources.get_distribution(lib).version

    return versions


def compare_package_versions(package_versions):
    """Compare the given package versions to the current package versions.

    Args:
        package_versions (dict[str, str]):
            A mapping from library to expected version.
    Returns:
        dict:
            A mapping of mismatched package versions.
    """
    mismatched_versions = {}
    for lib, version in package_versions.items():
        current_version = pkg_resources.get_distribution(lib).version
        if current_version != version:
            mismatched_versions[lib] = (version, current_version)

    return mismatched_versions
