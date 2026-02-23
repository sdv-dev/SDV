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
    """Test that the `download_demo` function works as intended for single-table."""
    # Run
    output_folder_name = tmp_path / 'sdv' if output_path else None
    data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests',
        output_folder_name=output_folder_name,
    )

    # Assert
    metadata.validate_data({'fake_hotel_guests': data})
    assert len(data) > 1
    assert isinstance(metadata, Metadata)


@pytest.mark.parametrize('output_path', [None, 'tmp_path'])
def test_download_demo_multi_table(output_path, tmp_path):
    """Test that the `download_demo` function works as intended for multi-table."""
    # Run
    output_folder_name = tmp_path / 'sdv' if output_path else None
    data, metadata = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels',
        output_folder_name=output_folder_name,
    )

    # Assert
    metadata.validate_data(data)
    expected_tables = ['hotels', 'guests']
    assert set(expected_tables) == set(data)
    assert isinstance(metadata, Metadata)
    assert len(data['hotels']) > 1
    assert len(data['guests']) > 1
