"""Miscellaneous utility functions."""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pkg_resources
from pandas.core.tools.datetimes import _guess_datetime_format_for_array


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

        names.append(f'<td style="text-align:left"><b>{name}</b></td>')
        data.append(f'<td>{table.head(max_rows).to_html(index=False)}</td>')

    if row:
        html = f"<table><tr>{''.join(names)}</tr><tr>{''.join(data)}</tr></table>"
    else:
        rows = [
            f'<tr>{name}</tr><tr>{table}</tr>'
            for name, table in zip(names, data)
        ]
        html = f"<table>{''.join(rows)}</table>"

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
    versions = {}
    try:
        versions['sdv'] = pkg_resources.get_distribution('sdv').version
        versions['rdt'] = pkg_resources.get_distribution('rdt').version
    except pkg_resources.ResolutionError:
        pass

    if model is not None:
        if not isinstance(model, type):
            model = model.__class__

        model_name = model.__module__ + model.__name__

        for lib in ['copulas', 'ctgan', 'deepecho']:
            if lib in model_name or ('hma' in model_name and lib == 'copulas'):
                try:
                    versions[lib] = pkg_resources.get_distribution(lib).version
                except pkg_resources.ResolutionError:
                    pass

    return versions


def throw_version_mismatch_warning(package_versions):
    """Throw mismatch warning if the given package versions don't match current package versions.

    If there is no mismatch, no warning is thrown.

    Args:
        package_versions (dict[str, str]):
            A mapping from library to expected version.

    Side Effects:
        A warning is thrown if there is a mismatch.
    """
    warning_str = (
        'The libraries used to create the model have older versions '
        'than your current setup. This may cause errors when sampling.'
    )

    if package_versions is None:
        warnings.warn(warning_str)
        return

    mismatched_details = ''
    for lib, version in package_versions.items():
        try:
            current_version = pkg_resources.get_distribution(lib).version
        except pkg_resources.ResolutionError:
            current_version = ''

        if current_version != version:
            mismatched_details += (
                f'\n{lib} used version `{version}`; '
                f'current version is `{current_version}`'
            )

    if len(mismatched_details) > 0:
        warnings.warn(f'{warning_str}{mismatched_details}')


def get_datetime_format(value):
    """Get the ``strftime`` format for a given ``value``.

    This function returns the ``strftime`` format of a given ``value`` when possible.
    If the ``_guess_datetime_format_for_array`` from ``pandas.core.tools.datetimes`` is
    able to detect the ``strftime`` it will return it as a ``string`` if not, a ``None``
    will be returned.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input to attempt detecting the format.

    Return:
        String representing the datetime format in ``strftime`` format or ``None`` if not detected.
    """
    if isinstance(value, pd.Series):
        value = value.astype(str).to_list()
    if not isinstance(value, (list, np.ndarray)):
        value = [value]

    return _guess_datetime_format_for_array(value)


def is_datetime_type(value):
    """Determine if the input is a datetime type or not.

    Args:
        value (pandas.DataFrame, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, (np.ndarray, pd.Series, list)):
        value = value[0]

    return (
        pd.api.types.is_datetime64_any_dtype(value)
        or isinstance(value, pd.Timestamp)
        or isinstance(value, datetime)
        or bool(get_datetime_format([value]))
    )


def is_numerical_type(value):
    """Determine if the input is numerical or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is numerical, False if not.
    """
    return pd.isna(value) | pd.api.types.is_float(value) | pd.api.types.is_integer(value)


def is_boolean_type(value):
    """Determine if the input is a boolean or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a boolean, False if not.
    """
    return True if pd.isna(value) | (value is True) | (value is False) else False
