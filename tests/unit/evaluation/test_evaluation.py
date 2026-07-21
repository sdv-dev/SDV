from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdmetrics.reports import DiagnosticReport, QualityReport

from sdv.evaluation.evaluation import (
    DEFAULT_SINGLE_TABLE_NAME,
    _handle_single_table,
    _validate_data,
    _validate_data_type,
    evaluate_quality,
    run_diagnostic,
)
from sdv.metadata.metadata import Metadata


@pytest.mark.parametrize(
    'data',
    [
        pd.DataFrame({'column': [1, 2]}),
        {'table': pd.DataFrame({'column': [1, 2]})},
    ],
)
def test__validate_data_type_accepts_allowed_types(data):
    """Test that `_validate_data_type` passes if the data is in the valid formats."""
    # Run and Assert
    _validate_data_type(data, 'real_data')


@pytest.mark.parametrize(
    ('data', 'type_name'),
    [
        (None, 'NoneType'),
        ([1, 2], 'list'),
        ('data', 'str'),
        (123, 'int'),
    ],
)
def test__validate_data_type_raises_for_invalid_type(data, type_name):
    """Test that `_validate_data_type` raises an error for an invalid data type."""
    # Setup
    expected_message = f'real_data must be a pandas DataFrame or dictionary, got {type_name}.'

    # Run
    with pytest.raises(TypeError) as error:
        _validate_data_type(data, 'real_data')

    # Assert
    assert str(error.value) == expected_message


@pytest.mark.parametrize(
    ('real_data', 'synthetic_data'),
    [
        (
            pd.DataFrame({'column': [1]}),
            pd.DataFrame({'column': [2]}),
        ),
        (
            {'table': pd.DataFrame({'column': [1]})},
            {'table': pd.DataFrame({'column': [2]})},
        ),
    ],
)
def test__validate_data_accepts_matching_types(real_data, synthetic_data):
    """Test that `_validate_data` passes when both inputs have matching valid types."""
    # Run and Assert
    _validate_data(real_data, synthetic_data)


def test__validate_data_raises_when_real_data_has_invalid_type():
    """Test that `_validate_data` raises an error when `real_data` is invalid."""
    # Setup
    real_data = []
    synthetic_data = pd.DataFrame({'column': [1]})
    expected_message = 'real_data must be a pandas DataFrame or dictionary, got list.'

    # Run
    with pytest.raises(TypeError) as error:
        _validate_data(real_data, synthetic_data)

    # Assert
    assert str(error.value) == expected_message


def test__validate_data_raises_when_synthetic_data_has_invalid_type():
    """Test that `_validate_data` raises an error when `synthetic_data` is invalid."""
    # Setup
    real_data = pd.DataFrame({'column': [1]})
    synthetic_data = []
    expected_message = 'synthetic_data must be a pandas DataFrame or dictionary, got list.'

    # Run
    with pytest.raises(TypeError) as error:
        _validate_data(real_data, synthetic_data)

    # Assert
    assert str(error.value) == expected_message


def test__validate_data_raises_when_inputs_have_different_types():
    """Test that `_validate_data` raises an error when input types do not match."""
    # Setup
    real_data = pd.DataFrame({'column': [1]})
    synthetic_data = {
        'table': pd.DataFrame({'column': [2]}),
    }
    expected_message = (
        'real_data and synthetic_data must have the same type. Got DataFrame and dict.'
    )

    # Run
    with pytest.raises(TypeError) as error:
        _validate_data(real_data, synthetic_data)

    # Assert
    assert str(error.value) == expected_message


def test__handle_single_table_wraps_dataframes_using_metadata_table_name():
    """Test that `_handle_single_table` uses the table name from the metadata."""
    # Setup
    real_data = pd.DataFrame({'column': [1, 2]})
    synthetic_data = pd.DataFrame({'column': [3, 4]})
    metadata = Mock(spec=Metadata)
    metadata._get_single_table_name.return_value = 'customers'

    # Run
    result_real, result_synthetic, result_metadata = _handle_single_table(
        real_data,
        synthetic_data,
        metadata,
    )

    # Assert
    assert result_real == {'customers': real_data}
    assert result_synthetic == {'customers': synthetic_data}
    assert result_metadata is metadata
    metadata._get_single_table_name.assert_called_once_with()


def test__handle_single_table_uses_default_name_when_metadata_has_no_table_name():
    """Test that `_handle_single_table` uses the default table name when none exists."""
    # Setup
    real_data = pd.DataFrame({'column': [1, 2]})
    synthetic_data = pd.DataFrame({'column': [3, 4]})
    metadata = Mock(spec=Metadata)
    metadata._get_single_table_name.return_value = None

    # Run
    result_real, result_synthetic, result_metadata = _handle_single_table(
        real_data,
        synthetic_data,
        metadata,
    )

    # Assert
    assert result_real == {DEFAULT_SINGLE_TABLE_NAME: real_data}
    assert result_synthetic == {DEFAULT_SINGLE_TABLE_NAME: synthetic_data}
    assert result_metadata is metadata
    metadata._get_single_table_name.assert_called_once_with()


@patch('sdv.evaluation.evaluation.Metadata.load_from_dict')
def test__handle_single_table_converts_non_metadata_object(mock_load_from_dict):
    """Test that `_handle_single_table` converts legacy metadata to `Metadata`."""
    # Setup
    real_data = pd.DataFrame({'column': [1, 2]})
    synthetic_data = pd.DataFrame({'column': [3, 4]})

    metadata = Mock()
    metadata_dict = {'columns': {'column': {'sdtype': 'numerical'}}}
    metadata.to_dict.return_value = metadata_dict

    converted_metadata = Mock(spec=Metadata)
    mock_load_from_dict.return_value = converted_metadata

    # Run
    result_real, result_synthetic, result_metadata = _handle_single_table(
        real_data,
        synthetic_data,
        metadata,
    )

    # Assert
    metadata.to_dict.assert_called_once_with()
    mock_load_from_dict.assert_called_once_with(
        metadata_dict,
        single_table_name=DEFAULT_SINGLE_TABLE_NAME,
    )
    assert result_real == {DEFAULT_SINGLE_TABLE_NAME: real_data}
    assert result_synthetic == {DEFAULT_SINGLE_TABLE_NAME: synthetic_data}
    assert result_metadata is converted_metadata


def test__handle_single_table_returns_dictionary_inputs_unchanged():
    """Test that `_handle_single_table` returns dictionary inputs unchanged."""
    # Setup
    real_data = {
        'customers': pd.DataFrame({'column': [1, 2]}),
    }
    synthetic_data = {
        'customers': pd.DataFrame({'column': [3, 4]}),
    }
    metadata = Mock(spec=Metadata)

    # Run
    result_real, result_synthetic, result_metadata = _handle_single_table(
        real_data,
        synthetic_data,
        metadata,
    )

    # Assert
    assert result_real is real_data
    assert result_synthetic is synthetic_data
    assert result_metadata is metadata
    metadata._get_single_table_name.assert_not_called()


def test__handle_single_table_returns_inputs_unchanged_when_types_do_not_match():
    """Test that `_handle_single_table` does nothing when only one input is a DataFrame."""
    # Setup
    real_data = pd.DataFrame({'column': [1]})
    synthetic_data = {
        'table': pd.DataFrame({'column': [2]}),
    }
    metadata = Mock(spec=Metadata)

    # Run
    result_real, result_synthetic, result_metadata = _handle_single_table(
        real_data,
        synthetic_data,
        metadata,
    )

    # Assert
    assert result_real is real_data
    assert result_synthetic is synthetic_data
    assert result_metadata is metadata
    metadata._get_single_table_name.assert_not_called()


@pytest.mark.parametrize(
    'data_format',
    [
        'dataframe',
        'dictionary',
    ],
)
def test_evaluate_quality_calls_generate(data_format):
    """Test that `evaluate_quality` calls `QualityReport.generate` for each data format."""
    # Setup
    real_table = pd.DataFrame({'col': [1, 2, 3]})
    synthetic_table = pd.DataFrame({'col': [2, 1, 3]})

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    if data_format == 'dataframe':
        real_data = real_table
        synthetic_data = synthetic_table
        expected_real_data = {'table': real_table}
        expected_synthetic_data = {'table': synthetic_table}
    else:
        real_data = {'table': real_table}
        synthetic_data = {'table': synthetic_table}
        expected_real_data = real_data
        expected_synthetic_data = synthetic_data

    quality_report = Mock(spec=QualityReport)

    # Run
    with patch(
        'sdv.evaluation.evaluation.QualityReport',
        return_value=quality_report,
    ):
        result = evaluate_quality(
            real_data,
            synthetic_data,
            metadata,
        )

    # Assert
    quality_report.generate.assert_called_once_with(
        expected_real_data,
        expected_synthetic_data,
        metadata.to_dict(),
        True,
    )
    assert result is quality_report


@pytest.mark.parametrize(
    'data_format',
    [
        'dataframe',
        'dictionary',
    ],
)
def test__run_diagnostic_calls_generate(data_format):
    """Test that `run_diagnostic` calls `DiagnosticReport.generate` for each data format."""
    # Setup
    real_table = pd.DataFrame({'col': [1, 2, 3]})
    synthetic_table = pd.DataFrame({'col': [2, 1, 3]})

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    if data_format == 'dataframe':
        real_data = real_table
        synthetic_data = synthetic_table
        expected_real_data = {'table': real_table}
        expected_synthetic_data = {'table': synthetic_table}
    else:
        real_data = {'table': real_table}
        synthetic_data = {'table': synthetic_table}
        expected_real_data = real_data
        expected_synthetic_data = synthetic_data

    diagnostic_report = Mock(spec=DiagnosticReport)

    # Run
    with patch(
        'sdv.evaluation.evaluation.DiagnosticReport',
        return_value=diagnostic_report,
    ):
        result = run_diagnostic(
            real_data,
            synthetic_data,
            metadata,
        )

    # Assert
    diagnostic_report.generate.assert_called_once_with(
        expected_real_data,
        expected_synthetic_data,
        metadata.to_dict(),
        True,
    )
    assert result is diagnostic_report
