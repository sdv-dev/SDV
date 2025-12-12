import pandas as pd
import pytest

from sdv.datasets.demo import download_demo, get_available_demos
from sdv.metadata import Metadata


def test_get_available_demos_single_table():
    """Test single_table demos listing equals the expected filtered list and values."""
    # Run
    tables_info = get_available_demos('single_table')
    mask = ~(
        tables_info['dataset_name'].str.startswith('bad_')
        | tables_info['dataset_name'].str.startswith('dataset')
    )
    tables_info = tables_info[mask].reset_index(drop=True)

    # Assert
    expected = pd.DataFrame({
        'dataset_name': [
            'adult',
            'alarm',
            'asia',
            'census',
            'census_extended',
            'child',
            'covtype',
            'expedia_hotel_logs',
            'fake_companies',
            'fake_hotel_guests',
            'insurance',
            'intrusion',
            'news',
            'student_placements',
            'student_placements_pii',
        ],
        'size_MB': [
            3.91,
            4.52,
            1.28,
            98.17,
            4.95,
            3.20,
            255.65,
            0.20,
            0.00,
            0.03,
            3.34,
            162.04,
            18.71,
            0.03,
            0.03,
        ],
        'num_tables': [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    })
    pd.testing.assert_frame_equal(tables_info[['dataset_name', 'size_MB', 'num_tables']], expected)


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
