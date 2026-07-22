import re

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo, get_available_demos, get_source, save_resource
from sdv.metadata import Metadata


def test_get_available_demos_single_table():
    """Test single_table demos listing is non-empty with valid sizes and table counts."""
    # Run
    tables_info = get_available_demos('single_table')

    # Assert
    assert not tables_info.empty
    assert (tables_info['num_tables'] == 1).all()
    assert (tables_info['size_MB'] >= 0).all()


def test_get_available_demos_multi_table():
    """Test multi_table demos listing is non-empty with valid sizes and table counts."""
    # Run
    tables_info = get_available_demos('multi_table')

    # Assert
    assert not tables_info.empty
    assert (tables_info['num_tables'] > 1).all()
    assert (tables_info['size_MB'] >= 0).all()


@pytest.mark.parametrize('output_path', [None, 'tmp_path'])
def test_download_demo_single_table(output_path, tmp_path):
    """Test `download_demo` function works for single-table."""
    # Setup
    output_folder_name = tmp_path / 'sdv' if output_path else None

    # Run
    data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests',
        output_folder_name=output_folder_name,
    )

    # Assert
    assert isinstance(metadata, Metadata)
    metadata.validate()
    assert isinstance(data, pd.DataFrame)
    metadata.validate_data({'fake_hotel_guests': data})
    assert len(data) > 1
    if output_folder_name:
        assert (output_folder_name / 'metadata.json').is_file()
        csv_files = list((output_folder_name / 'data').glob('*.csv'))
        assert len(csv_files) == 1
        assert csv_files[0].name == 'fake_hotel_guests.csv'


@pytest.mark.parametrize('output_path', [None, 'tmp_path'])
def test_download_demo_multi_table(output_path, tmp_path):
    """Test `download_demo` function works for multi-table."""
    # Setup
    output_folder_name = tmp_path / 'sdv' if output_path else None

    # Run
    data, metadata = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels',
        output_folder_name=output_folder_name,
    )

    # Assert
    assert isinstance(metadata, Metadata)
    metadata.validate()
    assert isinstance(data, dict)
    metadata.validate_data(data)
    expected_tables = ['hotels', 'guests']
    assert set(expected_tables) == set(data)
    assert len(data['hotels']) > 1
    assert len(data['guests']) > 1
    if output_folder_name is not None:
        assert (output_folder_name / 'metadata.json').is_file()
        csv_files = list((output_folder_name / 'data').glob('*.csv'))
        csv_files = [f.name for f in csv_files]
        assert len(csv_files) == 2
        assert 'hotels.csv' in csv_files
        assert 'guests.csv' in csv_files


@pytest.mark.parametrize('output_path', [None, 'tmp_path'])
def test_download_demo_sequential(output_path, tmp_path):
    """Test `download_demo` function works for sequential."""
    # Setup
    output_folder_name = tmp_path / 'sdv' if output_path else None

    # Run
    data, metadata = download_demo(
        modality='sequential',
        dataset_name='ArticularyWordRecognition',
        output_folder_name=output_folder_name,
    )

    # Assert
    assert isinstance(metadata, Metadata)
    metadata.validate()
    metadata = metadata._convert_to_single_table()
    metadata.validate_data(data)
    assert len(data) > 1
    if output_folder_name:
        assert (output_folder_name / 'metadata.json').is_file()
        csv_files = list((output_folder_name / 'data').glob('*.csv'))
        assert len(csv_files) == 1
        assert csv_files[0].name == 'ArticularyWordRecognition.csv'


def test_save_resource(tmp_path):
    """Test saving an arbitary demo resource."""
    # Setup
    modality = 'single_table'
    dataset_name = 'student_placements'
    resource_filepath = 'SOURCE.txt'
    output_filepath = tmp_path / 'SOURCE.txt'
    expected_source = get_source(modality, dataset_name)

    # Run
    save_resource(modality, dataset_name, resource_filepath, output_filepath)

    # Assert
    assert output_filepath.read_text() == expected_source


def test_save_resource_with_resource_filepath(tmp_path):
    """Test saving file nested in a folder."""
    # Setup
    modality = 'multi_table'
    dataset_name = 'synthea'
    resource_filepath = 'schemas/postgre.sql'
    output_filepath = tmp_path / 'postgre.sql'

    # Run
    save_resource(
        modality, dataset_name, resource_filepath=resource_filepath, output_filepath=output_filepath
    )

    # Assert
    assert 'synthea' in output_filepath.read_text()
    assert 'CREATE TABLE' in output_filepath.read_text()


def test_save_resource_raises_future_warning(tmp_path):
    """Test saving with the deprecated ``resource_filename`` parameter."""
    # Setup
    modality = 'single_table'
    dataset_name = 'student_placements'
    resource_filename = 'SOURCE.txt'
    output_filepath = tmp_path / 'SOURCE.txt'
    expected_source = get_source(modality, dataset_name)
    warning_msg = re.escape(
        'Warning: The `resource_filename` parameter is deprecated. '
        'Please use the `resource_filepath` parameter instead.'
    )

    # Run and Assert
    with pytest.warns(FutureWarning, match=warning_msg):
        save_resource(
            modality,
            dataset_name,
            resource_filename=resource_filename,
            output_filepath=output_filepath,
        )
    assert output_filepath.read_text() == expected_source


def test_save_resource_missing_resource_filepath():
    """Test error is raised if ``resource_filepath`` not provided."""
    # Setup
    error_msg = re.escape('Please provide a `resource_filepath`.')

    # Run and Assert
    with pytest.raises(ValueError, match=error_msg):
        save_resource(
            'single_table',
            'student_placements',
        )


def test_save_resource_missing_output_filepath():
    """Test error is raised if ``output_filepath`` not provided."""
    # Setup
    error_msg = re.escape('Please provide an `output_filepath`.')

    # Run and Assert
    with pytest.raises(ValueError, match=error_msg):
        save_resource('single_table', 'student_placements', 'SOURCE.txt')


def test_save_resource_both_resource_filepath_resource_filename():
    """Test error is raised if conflicting params provided."""
    # Setup
    error_msg = re.escape(
        'Cannot use both `resource_filepath` and `resource_filename`. '
        'Please use only `resource_filepath`.'
    )

    # Run and Assert
    with pytest.raises(ValueError, match=error_msg):
        save_resource(
            'single_table',
            'student_placements',
            resource_filepath='SOURCE.txt',
            output_filepath='SOURCE.txt',
            resource_filename='SOURCE.txt',
        )


def test_save_resource_resource_filepath_with_leading_slash(tmp_path):
    """Test a resource filepath with a leading slash errors."""
    # Setup
    error_msg = re.escape(
        "`resource_filepath` must be relative to the dataset and cannot begin with '/'."
    )

    # Run and Assert
    with pytest.raises(ValueError, match=error_msg):
        save_resource(
            modality='multi_table',
            dataset_name='synthea',
            resource_filepath='/schemas/postgre.sql',
            output_filepath=tmp_path / 'postgre.sql',
        )
