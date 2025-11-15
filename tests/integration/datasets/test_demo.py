import pandas as pd

from sdv.datasets.demo import get_available_demos


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
