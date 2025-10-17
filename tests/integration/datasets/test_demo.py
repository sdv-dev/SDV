import pandas as pd

from sdv.datasets.demo import get_available_demos, get_readme, get_source


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
        ],
    })
    pd.testing.assert_frame_equal(tables_info[['dataset_name', 'size_MB', 'num_tables']], expected)


def test_get_available_demos_multi_table():
    """Test multi_table demos listing is returned with expected columns and types."""
    # Run
    tables_info = get_available_demos('multi_table')

    # Assert
    expected = pd.DataFrame({
        'dataset_name': [
            'fake_hotels',
            'fake_hotels_extended',
        ],
        'size_MB': [
            0.05,
            0.07,
        ],
        'num_tables': [2, 2],
    })
    pd.testing.assert_frame_equal(tables_info[['dataset_name', 'size_MB', 'num_tables']], expected)


def test_get_readme_and_source_single_table_dataset1(tmp_path):
    """Test it returns the README and SOURCE for a single table dataset."""
    # Run
    readme = get_readme('single_table', 'dataset1')
    source = get_source('single_table', 'dataset1')

    # Assert
    assert isinstance(readme, str) and 'sample dataset' in readme.lower()
    assert isinstance(source, str) and source.strip() == 'unknown'

    readme_out = tmp_path / 'r.txt'
    source_out = tmp_path / 's.txt'
    readme2 = get_readme('single_table', 'dataset1', str(readme_out))
    source2 = get_source('single_table', 'dataset1', str(source_out))
    assert readme2 == readme
    assert source2 == source
    assert readme_out.read_text(encoding='utf-8').strip() == readme.strip()
    assert source_out.read_text(encoding='utf-8').strip() == source.strip()


def test_get_readme_missing_returns_none():
    """Test it returns None when the README/SOURCE is missing."""
    # Run and Assert
    assert get_readme('single_table', 'dataset2') is None
    assert get_source('single_table', 'dataset2') is None
