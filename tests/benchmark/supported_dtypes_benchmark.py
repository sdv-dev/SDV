"""Benchmark for supported data types."""

import contextlib
import logging
from copy import deepcopy
from functools import partialmethod

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tqdm import tqdm

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.benchmark.utils import get_previous_result

LOGGER = logging.getLogger(__name__)

SINGLE_COLUMN_PREDEFINED_CONSTRAINTS = {
    'Positive': {
        'constraint_class': 'Positive',
        'constraint_parameters': {'column_name': '', 'strict_boundaries': False},
    },
    'Negative': {
        'constraint_class': 'Negative',
        'constraint_parameters': {'column_name': '', 'strict_boundaries': False},
    },
    'ScalarInequality': {
        'constraint_class': 'ScalarInequality',
        'constraint_parameters': {'column_name': '', 'relation': '>=', 'value': 0},
    },
    'ScalarRange': {
        'constraint_class': 'ScalarRange',
        'constraint_parameters': {
            'column_name': '',
            'low_value': 0,
            'high_value': 1,
            'strict_boundaries': False,
        },
    },
    'FixedIncrements': {
        'constraint_class': 'FixedIncrements',
        'constraint_parameters': {
            'column_name': '',
            'increment_value': 1,
        },
    },
}

MULTI_COLUMN_PREDEFINED_CONSTRAINTS = {
    'FixedCombinations': {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': [],
        },
    },
    'Inequality': {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': '',
            'high_column_name': '',
            'strict_boundaries': False,
        },
    },
    'Range': {
        'constraint_class': 'Range',
        'constraint_parameters': {
            'low_column_name': '',
            'middle_column_name': '',
            'high_column_name': '',
            'strict_boundaries': False,
        },
    },
}


EXPECTED_METADATA_SDTYPES = {
    # Pandas
    'pd.Int8': 'numerical',
    'pd.Int16': 'numerical',
    'pd.Int32': 'numerical',
    'pd.Int64': 'numerical',
    'pd.UInt8': 'numerical',
    'pd.UInt16': 'numerical',
    'pd.UInt32': 'numerical',
    'pd.UInt64': 'numerical',
    'pd.Float32': 'numerical',
    'pd.Float64': 'numerical',
    'pd.datetime64': 'datetime',
    'pd.boolean': 'categorical',
    'pd.object': 'categorical',
    'pd.category': 'categorical',
    'pd.string': 'categorical',
    'pd.timedelta64': 'datetime',
    'pd.Period': 'datetime',
    'pd.Complex': 'numerical',
    # NumPy
    'np.int8': 'numerical',
    'np.int16': 'numerical',
    'np.int32': 'numerical',
    'np.int64': 'numerical',
    'np.uint8': 'numerical',
    'np.uint16': 'numerical',
    'np.uint32': 'numerical',
    'np.uint64': 'numerical',
    'np.float16': 'numerical',
    'np.float32': 'numerical',
    'np.float64': 'numerical',
    'np.complex64': 'numerical',
    'np.complex128': 'numerical',
    'np.datetime64': 'datetime',
    'np.timedelta64': 'datetime',
    'np.object': 'categorical',
    'np.bool': 'categorical',
    'np.string': 'categorical',
    'np.unicode': 'categorical',
}


PANDAS_DTYPES = {
    'pd.Int8': pd.DataFrame({'pd.Int8': pd.Series([1, 2, -3, None, 4, 5], dtype='Int8')}),
    'pd.Int16': pd.DataFrame({'pd.Int16': pd.Series([1, 2, -3, None, 4, 5], dtype='Int16')}),
    'pd.Int32': pd.DataFrame({'pd.Int32': pd.Series([1, 2, -3, None, 4, 5], dtype='Int32')}),
    'pd.Int64': pd.DataFrame({'pd.Int64': pd.Series([1, 2, -3, None, 4, 5], dtype='Int64')}),
    'pd.UInt8': pd.DataFrame({'pd.UInt8': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt8')}),
    'pd.UInt16': pd.DataFrame({'pd.UInt16': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt16')}),
    'pd.UInt32': pd.DataFrame({'pd.UInt32': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt32')}),
    'pd.UInt64': pd.DataFrame({'pd.UInt64': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt64')}),
    'pd.Float32': pd.DataFrame({
        'pd.Float32': pd.Series([1.1, 1.2, 1.3, 1.4, None], dtype='Float32')
    }),
    'pd.Float64': pd.DataFrame({
        'pd.Float64': pd.Series([1.1, 1.2, 1.3, 1.4, None], dtype='Float64')
    }),
    'pd.boolean': pd.DataFrame({
        'pd.boolean': pd.Series([True, False, None, True, False], dtype='boolean')
    }),
    'pd.object': pd.DataFrame({'pd.object': pd.Series(['A', 'B', None, 'C'], dtype='object')}),
    'pd.string': pd.DataFrame({'pd.string': pd.Series(['A', 'B', None, 'C'], dtype='string')}),
    'pd.category': pd.DataFrame({
        'pd.category': pd.Series(['A', 'B', None, 'D'], dtype='category')
    }),
    'pd.datetime64': pd.DataFrame({
        'pd.datetime64': pd.Series(pd.date_range('2023-01-01', periods=3), dtype='datetime64[ns]')
    }),
    'pd.timedelta64': pd.DataFrame({
        'pd.timedelta64': pd.Series(
            [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=3)],
            dtype='timedelta64[ns]',
        )
    }),
    'pd.Period': pd.DataFrame({
        'pd.Period': pd.Series(pd.period_range('2023-01', periods=3, freq='M')),
    }),
    'pd.Complex': pd.DataFrame({
        'pd.Complex': pd.Series([1 + 1j, 2 + 2j, 3 + 3j], dtype='complex128'),
    }),
}

NUMPY_DTYPES = {
    'np.int8': pd.DataFrame({
        'np.int8': pd.Series([np.int8(1), np.int8(-1), np.int8(127)], dtype='int8')
    }),
    'np.int16': pd.DataFrame({
        'np.int16': pd.Series([np.int16(2), np.int16(-2), np.int16(32767)], dtype='int16')
    }),
    'np.int32': pd.DataFrame({
        'np.int32': pd.Series([np.int32(3), np.int32(-3), np.int32(2147483647)], dtype='int32')
    }),
    'np.int64': pd.DataFrame({
        'np.int64': pd.Series([np.int64(4), np.int64(-4), np.int64(922)], dtype='int64')
    }),
    'np.uint8': pd.DataFrame({
        'np.uint8': pd.Series([np.uint8(5), np.uint8(10), np.uint8(255)], dtype='uint8')
    }),
    'np.uint16': pd.DataFrame({
        'np.uint16': pd.Series([np.uint16(6), np.uint16(20), np.uint16(65535)], dtype='uint16')
    }),
    'np.uint32': pd.DataFrame({
        'np.uint32': pd.Series([np.uint32(7), np.uint32(30), np.uint32(42)], dtype='uint32')
    }),
    'np.uint64': pd.DataFrame({
        'np.uint64': pd.Series([np.uint64(8), np.uint64(40), np.uint64(184467)], dtype='uint64')
    }),
    'np.float16': pd.DataFrame({
        'np.float16': pd.Series(
            [np.float16(9.1), np.float16(-9.1), np.float16(65.0)], dtype='float16'
        )
    }),
    'np.float32': pd.DataFrame({
        'np.float32': pd.Series(
            [np.float32(1.2), np.float32(-1.2), np.float32(3.40)], dtype='float32'
        )
    }),
    'np.float64': pd.DataFrame({
        'np.float64': pd.Series(
            [np.float64(1.3), np.float64(-11.3), np.float64(1.7)], dtype='float64'
        )
    }),
    'np.complex64': pd.DataFrame({
        'np.complex64': pd.Series(
            [np.complex64(12 + 1j), np.complex64(-12 - 1j), np.complex64(3.4e38 + 1j)],
            dtype='complex64',
        )
    }),
    'np.complex128': pd.DataFrame({
        'np.complex128': pd.Series(
            [np.complex128(13 + 2j), np.complex128(-13 - 2j), np.complex128(1.7e308 + 2j)],
            dtype='complex128',
        )
    }),
    'np.bool': pd.DataFrame({
        'np.bool': pd.Series([np.bool_(True), np.bool_(False), np.bool_(True)], dtype='bool')
    }),
    'np.object': pd.DataFrame({
        'np.object': pd.Series(['object1', 'object2', 'object3'], dtype='object')
    }),
    'np.string': pd.DataFrame({
        'np.string': pd.Series([
            np.string_('string1'),
            np.string_('string2'),
            np.string_('string3'),
        ])
    }),
    'np.unicode': pd.DataFrame({
        'np.unicode': pd.Series(
            [np.unicode_('unicode1'), np.unicode_('unicode2'), np.unicode_('unicode3')],
            dtype='string',
        )
    }),
    'np.datetime64': pd.DataFrame({
        'np.datetime64': pd.Series([
            np.datetime64('2023-01-01T00:00:00'),
            np.datetime64('2024-01-01T00:00:00'),
            np.datetime64('2025-01-01T00:00:00'),
        ])
    }),
    'np.timedelta64': pd.DataFrame({
        'np.timedelta64': pd.Series(
            [np.timedelta64(1, 'D'), np.timedelta64(2, 'h'), np.timedelta64(3, 'm')],
        )
    }),
}

PYARROW_DTYPES = {
    'pa.int8': pd.DataFrame({'pa.int8': pd.Series([1, -1, 127], dtype=pd.ArrowDtype(pa.int8()))}),
    'pa.int16': pd.DataFrame({
        'pa.int16': pd.Series([2, -2, 32767], dtype=pd.ArrowDtype(pa.int16()))
    }),
    'pa.int32': pd.DataFrame({
        'pa.int32': pd.Series([3, -3, 2147483647], dtype=pd.ArrowDtype(pa.int32()))
    }),
    'pa.int64': pd.DataFrame({
        'pa.int64': pd.Series([4, -4, 9223372036854775807], dtype=pd.ArrowDtype(pa.int64()))
    }),
    'pa.uint8': pd.DataFrame({
        'pa.uint8': pd.Series([5, 10, 255], dtype=pd.ArrowDtype(pa.uint8()))
    }),
    'pa.uint16': pd.DataFrame({
        'pa.uint16': pd.Series([6, 20, 65535], dtype=pd.ArrowDtype(pa.uint16()))
    }),
    'pa.uint32': pd.DataFrame({
        'pa.uint32': pd.Series([7, 30, 4294967295], dtype=pd.ArrowDtype(pa.uint32()))
    }),
    'pa.uint64': pd.DataFrame({
        'pa.uint64': pd.Series([8, 40, 18446744073709551615], dtype=pd.ArrowDtype(pa.uint64()))
    }),
    'pa.float32': pd.DataFrame({
        'pa.float32': pd.Series([1.2, -1.2, 3.40], dtype=pd.ArrowDtype(pa.float32()))
    }),
    'pa.float64': pd.DataFrame({
        'pa.float64': pd.Series([1.3, -11.3, 1.7], dtype=pd.ArrowDtype(pa.float64()))
    }),
    'pa.bool': pd.DataFrame({
        'pa.bool': pd.Series([True, False, True], dtype=pd.ArrowDtype(pa.bool_()))
    }),
    'pa.string': pd.DataFrame({
        'pa.string': pd.Series(['string1', 'string2', 'string3'], dtype=pd.ArrowDtype(pa.string()))
    }),
    'pa.date32': pd.DataFrame({
        'pa.date32': pd.Series(
            [pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01')],
            dtype=pd.ArrowDtype(pa.date32()),
        )
    }),
    'pa.timestamp': pd.DataFrame({
        'pa.timestamp': pd.Series(
            [
                pd.Timestamp('2023-01-01T00:00:00'),
                pd.Timestamp('2024-01-01T00:00:00'),
                pd.Timestamp('2025-01-01T00:00:00'),
            ],
            dtype=pd.ArrowDtype(pa.timestamp('ms')),
        )
    }),
    'pa.duration': pd.DataFrame({
        'pa.duration': pd.Series(
            [pd.Timedelta(days=1), pd.Timedelta(hours=2), pd.Timedelta(minutes=3)],
            dtype=pd.ArrowDtype(pa.duration('s')),
        )
    }),
    'pa.binary': pd.DataFrame({
        'pa.binary': pd.Series(
            [b'binary1', b'binary2', b'binary3'], dtype=pd.ArrowDtype(pa.binary())
        )
    }),
    'pa.utf8': pd.DataFrame({
        'pa.utf8': pd.Series(['utf8_1', 'utf8_2', 'utf8_3'], dtype=pd.ArrowDtype(pa.utf8()))
    }),
}


@contextlib.contextmanager
def prevent_tqdm_output():
    """Temporarily disables tqdm for the conditional sampling."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    try:
        yield
    finally:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def _get_metadata_for_dtype(dtype):
    """Return the expected metadata."""
    metadata = SingleTableMetadata.load_from_dict({
        'columns': {dtype: {'sdtype': EXPECTED_METADATA_SDTYPES.get(dtype, 'unknown')}}
    })
    return metadata


@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES}.items())
def test_metadata_detection(dtype, data):
    """Test metadata detection for data types using `SingleTableMetadata`.

    This test checks the ability of the `SingleTableMetadata` class to detect
    metadata from data types coming from `Pandas` and `NumPy`. It compares the
    detected metadata against expected results.

    Args:
        dtype (str):
            The data type to test.
        data (pd.DataFrame):
            The data for which metadata detection is performed.

    Raises:
        AssertionError:
            If the detected metadata is incorrect or the dtype is no longer supported.

    Test flow:
        1. Initialize `SingleTableMetadata`.
        2. Attempt to detect metadata from the provided data.
        3. Assert if the sdtype matches the expected one.
    """
    metadata = SingleTableMetadata()
    previous_result = get_previous_result(dtype, 'METADATA_DETECTION')
    result = False
    try:
        metadata.detect_from_dataframe(data)
        column = metadata.columns.get(dtype)
        sdtype = column.get('sdtype')
        result = sdtype == EXPECTED_METADATA_SDTYPES.get(dtype)
    except BaseException as e:
        LOGGER.debug(f"Error during 'metadata.validate_data' with dtype '{dtype}': {e}")

    assertion_message = f"{dtype} is no longer supported in 'METADATA_DETECTION'."
    if result is False:
        assert result == previous_result, assertion_message


@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES, **PYARROW_DTYPES}.items())
def test_metadata_validate_data(dtype, data):
    """Test the validation of data using `SingleTableMetadata`.

    This test checks whether the `validate_data` method of the metadata object
    properly validates the given data for different data types coming from
    `Pandas` and `NumPy`.

    Args:
        dtype (str):
            The data type to test.
        data (pd.DataFrame):
            The data for which metadata validation is performed.

    Raises:
        AssertionError:
            If the validation result does not match the previously recorded result
            or if the dtype is no longer supported.

    Test flow:
        1. Create a predefined `SingleTableMetadata` for the given dtype.
        2. Attempt to validate the data using `metadata.validate_data` for the provided data.
        3. Assert if the result is as expected.
    """
    metadata = _get_metadata_for_dtype(dtype)
    previous_result = get_previous_result(dtype, 'METADATA_VALIDATE_DATA')
    result = False
    try:
        metadata.validate_data(data)
        result = True
    except BaseException as e:
        LOGGER.debug(f"Error during 'metadata.validate_data' with dtype '{dtype}': {e}")

    if result is False:
        assertion_message = f"{dtype} is no longer supported by 'METADATA_VALIDATE_DATA'."
        assert result == previous_result, assertion_message


@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES, **PYARROW_DTYPES}.items())
def test_fit_and_sample_synthesizer(dtype, data):
    """Test fitting and sampling a synthesizer for different data types.

    This test evaluates the `GaussianCopulaSynthesizer` to fit and
    sample data for various data types from `Pandas` and `NumPy`.
    It verifies that the synthesizer can successfully be fitted to the
    data and generate synthetic data with matching data types.
    The results are compared with previously recorded outcomes for both
    fitting and sampling.

    Args:
        dtype (str):
            The data type to test.
        data (pd.DataFrame):
            The data for which the fitting and sampling is performed.

    Raises:
        AssertionError:
            If the fit or sample results do not match previously recorded results
            or if the dtype is no longer supported.

    The test flow includes:
        1. Initializing the `GaussianCopulaSynthesizer` with the appropriate metadata.
        2. Compare the current fit result against previously recorded results.
        3. Using the synthesizer to sample data, compare the synthetic data types if they match the
           input.
    """
    metadata = _get_metadata_for_dtype(dtype)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    previous_fit_result = get_previous_result(dtype, 'SYNTHESIZER_FIT')
    previous_sample_result = get_previous_result(dtype, 'SYNTHESIZER_SAMPLE')
    fit_result = False
    sample_result = False

    try:
        synthesizer.fit(data)
        fit_result = True
        with prevent_tqdm_output():
            synthetic_data = synthesizer.sample(10)

        sample_result = synthetic_data.dtypes[dtype] == data.dtypes[dtype]

    except BaseException as e:
        LOGGER.debug(f"Error during fitting/sampling with dtype '{dtype}': {e}")

    fit_assertion_message = f"{dtype} is no longer supported by 'SYNTHESIZER_FIT'."
    if fit_result is False:
        assert fit_result == previous_fit_result, fit_assertion_message

    sample_assertion_message = f"{dtype} is no longer supported by 'SYNTHESIZER_SAMPLE'."
    if sample_result is False:
        assert sample_result == previous_sample_result, sample_assertion_message


def convert_values(value, inequality):
    """Convert the given value based on the specified inequality.

    This function checks the provided value and applies a conversion based on
    the inequality function. If the value satisfies the inequality condition
    when compared to 0, it multiplies the value by -1. If the value is `None`,
    it returns `None`.

    Args:
        value (numeric):
            The value to be checked and potentially converted. It can be any numeric type or `None`.
        inequality (function):
            A comparison function (e.g., `operator.gt` or `operator.lt`) used to compare the
            value with 0.

    Returns:
        numeric or None:
            The converted value if the inequality holds, or the original value otherwise.
            Returns `None` if the value is `None`.
    """
    if pd.isna(value):
        return None

    if inequality(value, 0):
        return value * -1

    return value


def _create_single_column_constraint_and_data(constraint, data, dtype, sdtype):
    constraint_class = constraint.get('constraint_class')
    _dtype = data.dtypes[dtype]
    constraint['constraint_parameters']['column_name'] = dtype

    if constraint_class == 'Positive' and sdtype == 'numerical':
        data[dtype] = data[dtype].apply(convert_values, inequality=np.less)
    elif constraint_class == 'Negative' and sdtype == 'numerical':
        data[dtype] = data[dtype].apply(convert_values, inequality=np.greater)
    elif constraint_class == 'ScalarInequality':
        lower = 0
        if sdtype == 'numerical':
            data[dtype] = data[dtype].apply(convert_values, inequality=np.less)

        elif sdtype == 'datetime':
            # Make the lowest date to be 1971-01-01
            lower = '1971-01-01'

        constraint['constraint_parameters']['value'] = lower

    elif constraint_class == 'ScalarRange':
        if sdtype in ('numerical', 'datetime'):
            low_value = data[dtype].min()
            high_value = data[dtype].max()
            constraint['constraint_parameters']['low_value'] = low_value
            constraint['constraint_parameters']['high_value'] = high_value

    elif constraint_class == 'FixedIncrements':
        if sdtype == 'numerical':
            values = [10, 20, 30, 40]
            if dtype.startswith('pd'):
                values.append(None)

            data[dtype] = pd.Series(values, dtype=_dtype)
            constraint['constraint_parameters']['increment_value'] = 10

    return constraint, data


def _create_multi_column_constraint_data_and_metadata(constraint, data, dtype, sdtype, metadata):
    _dtype = data.dtypes[dtype]
    constraint_class = constraint.get('constraint_class')
    constraints = []
    if constraint_class == 'FixedCombinations':
        for dtype_name, dtype_data in {**PANDAS_DTYPES, **NUMPY_DTYPES}.items():
            dtype_sdtype = EXPECTED_METADATA_SDTYPES.get(dtype_name, 'unknown')
            if dtype_sdtype in ('categorical', 'boolean'):
                data[f'{dtype}_{dtype_name}'] = data[dtype]
                metadata.columns[f'{dtype}_{dtype_name}'] = {'sdtype': sdtype}
                new_constraint = deepcopy(constraint)
                data[dtype_name] = dtype_data[dtype_name]
                dtype_sdtype = EXPECTED_METADATA_SDTYPES.get(dtype_name, 'unknown')
                metadata.columns[dtype_name] = {'sdtype': dtype_sdtype}
                new_constraint['constraint_parameters']['column_names'].append(dtype_name)
                new_constraint['constraint_parameters']['column_names'].append(
                    f'{dtype}_{dtype_name}'
                )
                constraints.append(new_constraint)

    elif constraint_class == 'Inequality':
        if sdtype == 'numerical':
            data['high'] = data[dtype] * 10
            metadata.columns['high'] = {'sdtype': 'numerical'}

    elif constraint_class == 'Range':
        if sdtype == 'numerical':
            data['mid'] = data[dtype] * 5
            data['high'] = data[dtype] * 10
            metadata.columns['mid'] = {'sdtype': 'numerical'}
            metadata.columns['high'] = {'sdtype': 'numerical'}

    return constraints, data, metadata


@pytest.mark.parametrize(
    'constraint_name, constraint', SINGLE_COLUMN_PREDEFINED_CONSTRAINTS.items()
)
@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES, **PYARROW_DTYPES}.items())
def test_fit_and_sample_single_column_constraints(constraint_name, constraint, dtype, data):
    """Test fitting and sampling with single-column constraints for various data types.

    This test evaluates the `GaussianCopulaSynthesizer` to fit data and
    generate synthetic data while applying single-column constraints to different
    data types. It verifies that the synthesizer can respect the constraint and
    successfully produce synthetic data with the same data types as the original.

    Args:
        constraint_name (str):
            The name of the constraint being tested.
        constraint (dict):
            The predefined constraint to apply to the data.
        dtype (str):
            The data type being tested.
        data (pd.DataFrame):
            The input data to fit and generate synthetic samples from.

    Raises:
        AssertionError:
            If the fit or sample results do not match previously recorded results or if the dtype
            is no longer supported.

    The test flow includes:
        1. Initializing the `GaussianCopulaSynthesizer` with the metadata.
        2. Preparing the constraint and data for the test.
        3. Adding the constraint to the synthesizer, fitting the data, and verifying the fit result.
        4. Sampling synthetic data and checking that the dtype matches the original.
    """
    metadata = _get_metadata_for_dtype(dtype)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    sdtype = metadata.columns[dtype].get('sdtype')
    previous_fit_result = get_previous_result(dtype, f'{constraint_name}_FIT')
    previous_sample_result = get_previous_result(dtype, f'{constraint_name}_SAMPLE')

    # Prepare the constraint and data
    constraint, data = _create_single_column_constraint_and_data(
        deepcopy(constraint), data.copy(), dtype, sdtype
    )

    # Initialize results
    sample_result = False
    fit_result = False
    try:
        synthesizer.add_constraints([constraint])
        synthesizer.fit(data)
        fit_result = True

        # Sample Synthetic Data
        with prevent_tqdm_output():
            synthetic_data = synthesizer.sample(10)

        sample_result = synthetic_data.dtypes[dtype] == data.dtypes[dtype]

    except BaseException as e:
        LOGGER.debug(
            f"Error during fitting/sampling with dtype '{dtype}' and constraint "
            f"'{constraint_name}': {e}"
        )

    if fit_result is False:
        fit_assertion_message = f"{dtype} is no longer supported by '{constraint_name}_FIT''."
        assert fit_result == previous_fit_result, fit_assertion_message

    if sample_result is False:
        sample_assertion_message = f"{dtype} is no longer supported by '{constraint_name}_FIT''."
        assert sample_result == previous_sample_result, sample_assertion_message


@pytest.mark.parametrize('constraint_name, constraint', MULTI_COLUMN_PREDEFINED_CONSTRAINTS.items())
@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES, **PYARROW_DTYPES}.items())
def test_fit_and_sample_multi_column_constraints(constraint_name, constraint, dtype, data):
    """Test fitting and sampling with multi-column constraints for various data types.

    This test evaluates the `GaussianCopulaSynthesizer` to fit data and
    generate synthetic data while applying multi-column constraints. It ensures
    that the synthesizer can handle constraints across multiple columns and produce
    synthetic data with the expected data types.

    Args:
        constraint_name (str):
            The name of the multi-column constraint being tested.
        constraint (dict):
            The predefined multi-column constraint to apply to the data.
        dtype (str):
            The data type being tested.
        data (pd.DataFrame):
            The input data to fit and generate synthetic samples from.

    Raises:
        AssertionError:
            If the fit or sample results do not match previously recorded results or if
            the dtype is no longer supported.

    The test flow includes:
        1. Preparing the constraints, data, and metadata for the test.
        2. Initializing the `GaussianCopulaSynthesizer` with the metadata.
        3. Adding the multi-column constraints to the synthesizer and fitting the data.
        4. Sampling synthetic data and ensuring the synthetic data types match the original.
    """

    metadata = _get_metadata_for_dtype(dtype)
    sdtype = metadata.columns[dtype].get('sdtype')
    previous_fit_result = get_previous_result(dtype, f'{constraint_name}_FIT')
    previous_sample_result = get_previous_result(dtype, f'{constraint_name}_SAMPLE')

    # Prepare constraints, data required and metadata
    constraints, data, metadata = _create_multi_column_constraint_data_and_metadata(
        deepcopy(constraint), data.copy(), dtype, sdtype, metadata
    )

    # Initialize results
    sample_result = False
    fit_result = False

    try:
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.add_constraints(constraints)
        synthesizer.fit(data)
        fit_result = True

        # Generate Synthetic Data
        with prevent_tqdm_output():
            synthetic_data = synthesizer.sample(10)

        sample_result = synthetic_data.dtypes[dtype] == data.dtypes[dtype]

    except BaseException as e:
        LOGGER.debug(
            f"Error during fitting/sampling with dtype '{dtype}' and constraint "
            f"'{constraint_name}': {e}"
        )

    # Assertions - Only if they are False
    if fit_result is False:
        assert fit_result == previous_fit_result, f"{dtype} failed during '{constraint_name}_FIT'."

    if sample_result is False:
        sample_msg = f"{dtype} failed during '{constraint_name}_SAMPLE'."
        assert sample_result == previous_sample_result, sample_msg
