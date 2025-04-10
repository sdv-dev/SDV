"""Miscellaneous utility functions."""

import inspect
import operator
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import is_float, is_integer
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
from rdt.transformers.utils import _GENERATORS, strings_from_regex

from sdv import version
from sdv.errors import SDVVersionWarning, SynthesizerInputError, VersionError

try:
    from re import _parser as sre_parse
except ImportError:
    import sre_parse


def _cast_to_iterable(value):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        return value

    return [value]


def _get_datetime_format(value):
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
    if not isinstance(value, pd.Series):
        value = pd.Series(value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()

    return _guess_datetime_format_for_array(value)


def _is_datetime_type(value):
    """Determine if the input is a datetime type or not.

    If a ``pandas.Series`` or ``list`` is passed, it will return ``True`` if the first
    thousand values are datetime. Otherwise, it will check if the value is a datetime.

    Note: it will return ``False`` if ``value`` is a string representing
    a date before the year 1677.

    Args:
        value (array-like iterable, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, str) or (not isinstance(value, Iterable)):
        value = _cast_to_iterable(value)

    values = pd.Series(value)
    values = values[~values.isna()]
    values = values.head(1000)  # only check 1000 values so this method takes less than 1 second
    for value in values:
        if not (
            bool(_get_datetime_format([value]))
            or isinstance(value, pd.Timestamp)
            or isinstance(value, datetime)
            or isinstance(value, pd.Period)
            or (isinstance(value, str) and pd.notna(pd.to_datetime(value, errors='coerce')))
        ):
            return False

    return True


def _is_numerical_type(value):
    """Determine if the input is numerical or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is numerical, False if not.
    """
    return pd.isna(value) | pd.api.types.is_float(value) | pd.api.types.is_integer(value)


def _is_boolean_type(value):
    """Determine if the input is a boolean or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a boolean, False if not.
    """
    return True if pd.isna(value) | (value is True) | (value is False) else False


def _validate_datetime_format(column, datetime_format):
    """Determine the values of the column that match the datetime format.

    Args:
        column (pd.Series):
            Column to evaluate.
        datetime_format (str):
            The datetime format.

    Returns:
        pd.Series:
            Series of booleans, with True if the value matches the format, False if not.
    """
    pandas_datetime_format = datetime_format.replace('%-', '%')
    datetime_column = pd.to_datetime(column, errors='coerce', format=pandas_datetime_format)
    valid = pd.isna(column) | ~pd.isna(datetime_column)

    return set(column[~valid])


def _convert_to_timedelta(column):
    """Convert a ``pandas.Series`` to one with dtype ``timedelta``.

    ``pd.to_timedelta`` does not handle nans, so this function masks the nans, converts and then
    reinserts them.

    Args:
        column (pandas.Series):
            Column to convert.

    Returns:
        pandas.Series:
            The column converted to timedeltas.
    """
    nan_mask = pd.isna(column)
    column[nan_mask] = 0
    column = pd.to_timedelta(column)
    column[nan_mask] = pd.NaT
    return column


def _load_data_from_csv(filepath, read_csv_parameters=None):
    """Load DataFrame from a filepath.

    Args:
        filepath (str):
            String that represents the ``path`` to the ``csv`` file.
        read_csv_parameters (dict):
            A python dictionary of with string and value accepted by ``pandas.read_csv``
            function. Defaults to ``None``.
    """
    filepath = Path(filepath)
    read_csv_parameters = read_csv_parameters or {}
    data = pd.read_csv(filepath, **read_csv_parameters)
    return data


def _groupby_list(list_to_check):
    """Return the first element of the list if the length is 1 else the entire list."""
    return list_to_check[0] if len(list_to_check) == 1 else list_to_check


def _create_unique_name(name, list_names):
    """Modify the ``name`` parameter if it already exists in the list of names."""
    result = name
    while result in list_names:
        result += '_'

    return result


def _format_invalid_values_string(invalid_values, num_values):
    """Convert ``invalid_values`` into a string of invalid values.

    Args:
        invalid_values (pd.DataFrame, set):
            Object of values to be converted into string.
        num_values (int):
            Maximum number of values of the object to show.

    Returns:
        str:
            A stringified version of the object.
    """
    if isinstance(invalid_values, pd.DataFrame):
        if len(invalid_values) > num_values:
            return f'{invalid_values.head(num_values)}\n+{len(invalid_values) - num_values} more'

    if isinstance(invalid_values, set):
        invalid_values = sorted(invalid_values, key=lambda x: str(x))
        if len(invalid_values) > num_values:
            extra_missing_values = [f'+ {len(invalid_values) - num_values} more']
            return f'{invalid_values[:num_values] + extra_missing_values}'

    return f'{invalid_values}'


def _validate_foreign_keys_not_null(metadata, data):
    """Validate that the foreign keys in the data don't have null values."""
    invalid_tables = defaultdict(list)
    for table_name, table_data in data.items():
        for foreign_key in metadata._get_all_foreign_keys(table_name):
            if foreign_key not in table_data and int(foreign_key) in table_data:
                foreign_key = int(foreign_key)
            if table_data[foreign_key].isna().any():
                invalid_tables[table_name].append(foreign_key)

    if invalid_tables:
        err_msg = (
            'The data contains null values in foreign key columns. '
            'This feature is currently unsupported. Please remove '
            'null values to fit the synthesizer.\n'
            '\n'
            'Affected columns:\n'
        )
        for table_name, invalid_columns in invalid_tables.items():
            err_msg += f"Table '{table_name}', column(s) {invalid_columns}\n"

        raise SynthesizerInputError(err_msg)


def check_sdv_versions_and_warn(synthesizer):
    """Check if the current SDV and SDV Enterprise versions mismatch.

    Args:
        synthesizer (BaseSynthesizer or BaseMultiTableSynthesizer):
            An SDV model instance to check versions against.

    Raises:
        SDVVersionWarning:
            If the current SDV or SDV Enterprise version does not match the version used to fit
            the synthesizer.
    """
    current_public_version = getattr(version, 'public', None)
    current_enterprise_version = getattr(version, 'enterprise', None)
    if synthesizer._fitted:
        fitted_public_version = getattr(synthesizer, '_fitted_sdv_version', None)
        fitted_enterprise_version = getattr(synthesizer, '_fitted_sdv_enterprise_version', None)
        public_missmatch = current_public_version != fitted_public_version
        enterprise_missmatch = current_enterprise_version != fitted_enterprise_version

        if public_missmatch or enterprise_missmatch:
            static_message = (
                'The latest bug fixes and features may not be available for this synthesizer. '
                'To see these enhancements, create and train a new synthesizer on this version.'
            )
            if public_missmatch and enterprise_missmatch:
                message = (
                    'You are currently on SDV version '
                    f'{current_public_version} and SDV Enterprise version '
                    f'{current_enterprise_version} but this synthesizer was created on '
                    f'SDV version {synthesizer._fitted_sdv_version} and SDV Enterprise version '
                    f'{synthesizer._fitted_sdv_enterprise_version}.'
                )

            elif public_missmatch:
                message = (
                    'You are currently on SDV version '
                    f'{current_public_version} but this synthesizer was created on '
                    f'version {synthesizer._fitted_sdv_version}.'
                )
            elif enterprise_missmatch:
                message = (
                    'You are currently on SDV Enterprise version '
                    f'{current_enterprise_version} but this synthesizer was created on '
                    f'version {synthesizer._fitted_sdv_enterprise_version}.'
                )

            message = f'{message} {static_message}'
            warnings.warn(message, SDVVersionWarning)


def _compare_versions(current_version, synthesizer_version, compare_operator=operator.gt):
    """Compare two versions.

    Given a ``compare_operator`` compare two versions using that operator to determine if one is
    greater than the other or vice-versa.

    Args:
        current_version (str):
            The current version to compare against, formatted as a string with major, minor, and
            revision parts separated by periods (e.g., "1.0.0").
        synthesizer_version (str):
            The synthesizer version to compare, formatted as a string with major, minor, and
            revision parts separated by periods (e.g., "1.0.0")
        compare_operator (operator):
            Operator function to evaluate with. Defaults to ``operator.gt``.

    Returns:
        bool:
            Depending on the ``operator`` function it will return ``True`` or ``False`` if
            ``current_version`` is bigger or lower than ``synthesizer_version``.
    """
    if None in (current_version, synthesizer_version):
        return False

    current_version = current_version.split('.')
    synthesizer_version = synthesizer_version.split('.')
    for current_v, synth_v in zip(current_version, synthesizer_version):
        try:
            current_v = int(current_v)
            synth_v = int(synth_v)
            if compare_operator(current_v, synth_v):
                return False

            if compare_operator(synth_v, current_v):
                return True

        except Exception:
            pass

    return False


def check_synthesizer_version(synthesizer, is_fit_method=False, compare_operator=operator.gt):
    """Check if the current synthesizer version is greater than the package version.

    Args:
        synthesizer (BaseSynthesizer or BaseMultiTableSynthesizer):
            An SDV model instance to check versions against.
        is_fit_method (bool):
            Whether or not this function is being called by a ``fit`` function.
        compare_operator (operator):
            Operator function to evaluate with. Defaults to ``operator.gt``.

    Raises:
        VersionError:
            If the current version of the software is lower than the synthesizer's version.
    """
    current_public_version = getattr(version, 'public', None)
    current_enterprise_version = getattr(version, 'enterprise', None)
    static_message = 'Downgrading your SDV version is not supported.'
    if is_fit_method:
        static_message = (
            'Fitting this synthesizer again is not supported. Please create a new synthesizer.'
        )

    fit_public_version = getattr(synthesizer, '_fitted_sdv_version', None)
    fit_enterprise_version = getattr(synthesizer, '_fitted_sdv_enterprise_version', None)

    is_public_lower = _compare_versions(
        current_public_version, fit_public_version, compare_operator
    )

    is_enterprise_lower = _compare_versions(
        current_enterprise_version, fit_enterprise_version, compare_operator
    )

    if is_public_lower and is_enterprise_lower:
        raise VersionError(
            f'You are currently on SDV version {current_public_version} and SDV Enterprise '
            f'version {current_enterprise_version} but this '
            f'synthesizer was created on SDV version {fit_public_version} and SDV '
            f'Enterprise version {fit_enterprise_version}. {static_message}'
        )

    if is_public_lower:
        raise VersionError(
            f'You are currently on SDV version {current_public_version} but this '
            f'synthesizer was created on version {fit_public_version}. {static_message}'
        )

    if is_enterprise_lower:
        raise VersionError(
            f'You are currently on SDV Enterprise version {current_enterprise_version} but '
            f'this synthesizer was created on version {fit_enterprise_version}. '
            f'{static_message}'
        )


def _get_root_tables(relationships):
    parent_tables = {rel['parent_table_name'] for rel in relationships}
    child_tables = {rel['child_table_name'] for rel in relationships}
    return parent_tables - child_tables


def generate_synthesizer_id(synthesizer):
    """Generate a unique identifier for the synthesizer instance.

    This method creates a unique identifier by combining the class name, the public SDV version
    and the last part of a UUID4 composed by 36 random characters.

    Args:
        synthesizer (BaseSynthesizer or BaseMultiTableSynthesizer):
            An SDV model instance to check versions against.

    Returns:
        ID:
            A unique identifier for this synthesizer.
    """
    class_name = synthesizer.__class__.__name__
    synth_version = version.public
    unique_id = ''.join(str(uuid.uuid4()).split('-'))
    return f'{class_name}_{synth_version}_{unique_id}'


def _get_chars_for_option(option, params):
    if option not in _GENERATORS:
        raise ValueError(f'REGEX operation: {option} is not supported by SDV.')

    if option == sre_parse.MAX_REPEAT:
        new_option, new_params = params[2][0]  # The value at the second index is the nested option
        return _get_chars_for_option(new_option, new_params)

    return list(_GENERATORS[option](params, 1)[0])


def get_possible_chars(regex, num_subpatterns=None):
    """Get the list of possible characters a regex can create.

    Args:
        regex (str):
            The regex to parse.
        num_subpatterns (int):
            The number of sub-patterns from the regex to find characters for.
    """
    parsed = sre_parse.parse(regex)
    parsed = [p for p in parsed if p[0] != sre_parse.AT]
    num_subpatterns = num_subpatterns or len(parsed)
    possible_chars = []
    for option, params in parsed[:num_subpatterns]:
        possible_chars += _get_chars_for_option(option, params)

    return possible_chars


def _is_numerical(value):
    """Determine if the input is a numerical type or not."""
    try:
        return is_integer(value) or is_float(value)
    except Exception:
        return False


def _get_transformer_init_kwargs(transformer):
    """Get the dict of arguments used to instantiate the given transformer."""
    args = inspect.getfullargspec(transformer.__init__).args[1:]
    return {
        key: getattr(transformer, key)
        for key in args
        if key != 'model_missing_values' and hasattr(transformer, key)
    }


def _check_regex_format(table_name, column_name, regex):
    """Check if SDV can generate data for the given regex."""
    if regex:
        try:
            strings_from_regex(regex)
        except Exception as e:
            raise SynthesizerInputError(
                "SDV synthesizers do not currently support complex regex formats such as '"
                f"{regex}', which you have provided for table '{table_name}', column '{column_name}"
                "'. Please use a simplified format or update to a different sdtype."
            ) from e
