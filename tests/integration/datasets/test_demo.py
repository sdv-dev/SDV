import pandas as pd
import pytest

from sdv.datasets.demo import download_demo, get_available_demos
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
    """Test that the `download_demo` function works for single-table."""
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
    """Test that the `download_demo` function works for multi-table."""
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
        for table_name in expected_tables:
            csv_path = output_folder_name / 'data' / f'{table_name}.csv'
            assert csv_path.is_file()
