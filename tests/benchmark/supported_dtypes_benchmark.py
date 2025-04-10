"""Benchmark for supported data types."""

import contextlib
import logging
from copy import deepcopy
from functools import partialmethod

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import (
    BinaryEncoder,
    ClusterBasedNormalizer,
    FloatFormatter,
    FrequencyEncoder,
    GaussianNormalizer,
    IDGenerator,
    LabelEncoder,
    LogitScaler,
    LogScaler,
    OneHotEncoder,
    OptimizedTimestampEncoder,
    RegexGenerator,
    UnixTimestampEncoder,
)
from tqdm import tqdm

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.benchmark.excluded_tests import EXCLUDED_DATA_TYPES
from tests.benchmark.included_tests import INCLUDED_CONSTRAINT_TESTS
from tests.benchmark.numpy_dtypes import NUMPY_DTYPES
from tests.benchmark.pandas_dtypes import PANDAS_DTYPES
from tests.benchmark.utils import get_previous_dtype_result, save_results_to_json

LOGGER = logging.getLogger(__name__)

SINGLE_COLUMN_PREDEFINED_CONSTRAINTS = {
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


METADATA_SDTYPES = (
    {'sdtype': 'numerical'},
    {'sdtype': 'id'},
    {'sdtype': 'datetime', 'datetime_format': '%Y%m%d'},
    {'sdtype': 'categorical'},
)


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
    'np.bytes': 'categorical',
    'np.unicode': 'categorical',
    # PyArrow
    'pa.int8': 'numerical',
    'pa.int16': 'numerical',
    'pa.int32': 'numerical',
    'pa.int64': 'numerical',
    'pa.uint8': 'numerical',
    'pa.uint16': 'numerical',
    'pa.uint32': 'numerical',
    'pa.uint64': 'numerical',
    'pa.float32': 'numerical',
    'pa.float64': 'numerical',
    'pa.bool': 'categorical',
    'pa.string': 'categorical',
    'pa.utf8': 'categorical',
    'pa.binary': 'categorical',
    'pa.large_binary': 'categorical',
    'pa.large_string': 'categorical',
    'pa.binary_view': 'categorical',
    'pa.string_view': 'categorical',
    'pa.date32': 'datetime',
    'pa.date64': 'datetime',
    'pa.timestamp': 'datetime',
    'pa.duration': 'datetime',
    'pa.time32': 'datetime',
    'pa.time64': 'datetime',
    'pa.decimal128': 'numerical',
}

TRANSFORMERS = {
    BinaryEncoder: {},
    ClusterBasedNormalizer: {'max_clusters': 3},
    FloatFormatter: {},
    FrequencyEncoder: {},
    GaussianNormalizer: {},
    IDGenerator: {},
    LabelEncoder: {},
    LogScaler: {'constant': np.iinfo(np.int8).min},
    LogitScaler: {'min_value': np.iinfo(np.int64).min, 'max_value': np.iinfo(np.int64).max},
    OneHotEncoder: {},
    OptimizedTimestampEncoder: {},
    RegexGenerator: {},
    UnixTimestampEncoder: {},
}


@contextlib.contextmanager
def prevent_tqdm_output():
    """Temporarily disables tqdm for the conditional sampling."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    try:
        yield
    finally:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def _get_metadata_for_dtype_and_sdtype(dtype, sdtype_dict):
    """Return the expected metadata."""
    metadata = SingleTableMetadata.load_from_dict({'columns': {dtype: sdtype_dict}})
    return metadata


def skip_if_unsupported(dtype, sdtype):
    if (dtype, sdtype) in EXCLUDED_DATA_TYPES:
        pytest.skip(f"unsupported data type combination '{dtype}' and '{sdtype}'")


@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES}.items())
@pytest.mark.parametrize('sdtype_dict', METADATA_SDTYPES)
def test_fit_and_sample_synthesizer(dtype, data, sdtype_dict):
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
    sdtype = sdtype_dict['sdtype']
    skip_if_unsupported(dtype, sdtype)

    metadata = _get_metadata_for_dtype_and_sdtype(dtype, sdtype_dict)
    synthesizer = GaussianCopulaSynthesizer(metadata)

    previous_fit_result, _ = get_previous_dtype_result(dtype, sdtype, 'SYNTHESIZER_FIT')
    previous_sample_result, _ = get_previous_dtype_result(dtype, sdtype, 'SYNTHESIZER_SAMPLE')
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

    save_results_to_json({
        'dtype': dtype,
        'sdtype': sdtype,
        'SYNTHESIZER_FIT': fit_result,
        'SYNTHESIZER_SAMPLE': sample_result,
    })
    fit_assertion_message = f"{dtype} is no longer supported by 'SYNTHESIZER_FIT'."
    if fit_result is False:
        assert fit_result == previous_fit_result, fit_assertion_message

    sample_assertion_message = f"{dtype} is no longer supported by 'SYNTHESIZER_SAMPLE'."
    if sample_result is False:
        assert sample_result == previous_sample_result, sample_assertion_message


@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES}.items())
@pytest.mark.parametrize('sdtype_dict', METADATA_SDTYPES)
@pytest.mark.parametrize('transformer, transformer_kwargs', TRANSFORMERS.items())
def test_transformer(dtype, data, sdtype_dict, transformer, transformer_kwargs):
    """Test using a transformer with different data types."""
    sdtype = sdtype_dict['sdtype']
    skip_if_unsupported(dtype, sdtype)

    _transformer = transformer(**transformer_kwargs)
    transformer_name = _transformer.get_name()

    if sdtype != transformer.INPUT_SDTYPE:
        pytest.skip("Sdtype does not match transformer's input type, skipping.")

    previous_fit_result, _ = get_previous_dtype_result(dtype, sdtype, f'RDT_{transformer_name}_FIT')
    previous_transform_result, _ = get_previous_dtype_result(
        dtype, sdtype, f'RDT_{transformer_name}_TRANSFORM'
    )
    previous_reverse_result, _ = get_previous_dtype_result(
        dtype, sdtype, f'RDT_{transformer_name}_REVERSE'
    )
    fit_result = False
    transform_result = False
    reverse_result = False

    try:
        _transformer.fit(data, dtype)
        fit_result = True
        transformed_data = _transformer.transform(data)
        transform_result = True
        _transformer.reverse_transform(transformed_data)
        reverse_result = True  # data == reversed_data instead?
    except BaseException as e:
        LOGGER.debug(f"Error during fitting/transform/reverse with dtype '{dtype}': {e}")

    save_results_to_json({
        'dtype': dtype,
        'sdtype': sdtype,
        f'RDT_{transformer_name}_FIT': fit_result,
        f'RDT_{transformer_name}_TRANSFORM': transform_result,
        f'RDT_{transformer_name}_REVERSE': reverse_result,
    })

    fit_assertion_message = f"{dtype} is no longer supported by 'RDT_{transformer_name}_FIT'."
    if fit_result is False:
        assert fit_result == previous_fit_result, fit_assertion_message

    transform_assertion_message = (
        f"{dtype} is no longer supported by 'RDT_{transformer_name}_TRANSFORM'."
    )
    if transform_result is False:
        assert transform_result == previous_transform_result, transform_assertion_message

    reverse_assertion_message = (
        f"{dtype} is no longer supported by 'RDT_{transformer_name}_REVERSE'."
    )
    if reverse_result is False:
        assert reverse_result == previous_reverse_result, reverse_assertion_message


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
@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES}.items())
@pytest.mark.parametrize('sdtype_dict', METADATA_SDTYPES)
def test_fit_and_sample_single_column_constraints(
    constraint_name, constraint, dtype, data, sdtype_dict
):
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
    sdtype = sdtype_dict['sdtype']
    skip_if_unsupported(dtype, sdtype)

    if (sdtype, constraint_name) in INCLUDED_CONSTRAINT_TESTS:
        metadata = _get_metadata_for_dtype_and_sdtype(dtype, sdtype_dict)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        sdtype = metadata.columns[dtype].get('sdtype')
        previous_fit_result, _ = get_previous_dtype_result(
            dtype, sdtype, f'CONSTRAINT_{constraint_name}_FIT'
        )
        previous_sample_result, _ = get_previous_dtype_result(
            dtype, sdtype, f'CONSTRAINT_{constraint_name}_SAMPLE'
        )

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

        save_results_to_json({
            'dtype': dtype,
            'sdtype': sdtype,
            f'CONSTRAINT_{constraint_name}_FIT': fit_result,
            f'CONSTRAINT_{constraint_name}_SAMPLE': sample_result,
        })
        if fit_result is False:
            fit_assertion_message = (
                f"{dtype} is no longer supported by 'CONSTRAINT_{constraint_name}_FIT''."
            )
            assert fit_result == previous_fit_result, fit_assertion_message

        if sample_result is False:
            sample_assertion_message = (
                f"{dtype} is no longer supported by 'CONSTRAINT_{constraint_name}_FIT''."
            )
            assert sample_result == previous_sample_result, sample_assertion_message


@pytest.mark.parametrize('constraint_name, constraint', MULTI_COLUMN_PREDEFINED_CONSTRAINTS.items())
@pytest.mark.parametrize('dtype, data', {**PANDAS_DTYPES, **NUMPY_DTYPES}.items())
@pytest.mark.parametrize('sdtype_dict', METADATA_SDTYPES)
def test_fit_and_sample_multi_column_constraints(
    constraint_name, constraint, dtype, data, sdtype_dict
):
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
    sdtype = sdtype_dict['sdtype']
    skip_if_unsupported(dtype, sdtype)

    if (sdtype, constraint_name) in INCLUDED_CONSTRAINT_TESTS:
        metadata = _get_metadata_for_dtype_and_sdtype(dtype, sdtype_dict)
        sdtype = metadata.columns[dtype].get('sdtype')
        previous_fit_result, _ = get_previous_dtype_result(
            dtype, sdtype, f'CONSTRAINT_{constraint_name}_FIT'
        )
        previous_sample_result, _ = get_previous_dtype_result(
            dtype, sdtype, f'CONSTRAINT_{constraint_name}_SAMPLE'
        )

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

        save_results_to_json({
            'dtype': dtype,
            'sdtype': sdtype,
            f'CONSTRAINT_{constraint_name}_FIT': fit_result,
            f'CONSTRAINT_{constraint_name}_SAMPLE': sample_result,
        })
        if fit_result is False:
            fit_message = f"{dtype} failed during 'CONSTRAINT_{constraint_name}_FIT'."
            assert fit_result == previous_fit_result, fit_message

        if sample_result is False:
            sample_msg = f"{dtype} failed during 'CONSTRAINT_{constraint_name}_SAMPLE'."
            assert sample_result == previous_sample_result, sample_msg
