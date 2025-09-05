from pandas.api.types import is_numeric_dtype

from sdv.datasets.demo import get_available_demos


def test_get_available_demos_single_table():
    """Test single_table demos listing is returned with expected columns and types."""
    tables_info = get_available_demos('single_table')

    assert set(['dataset_name', 'size_MB', 'num_tables']).issubset(tables_info.columns)
    assert len(tables_info) >= 1

    assert is_numeric_dtype(tables_info['size_MB'])
    assert is_numeric_dtype(tables_info['num_tables'])

    non_null = tables_info['num_tables'].dropna()
    assert all(float(x).is_integer() for x in non_null)

    names = set(tables_info['dataset_name'].astype(str).tolist())
    assert {'dataset1', 'dataset2'}.issubset(names)


def test_get_available_demos_multi_table():
    """Test multi_table demos listing is returned with expected columns and types."""
    tables_info = get_available_demos('multi_table')

    assert set(['dataset_name', 'size_MB', 'num_tables']).issubset(tables_info.columns)

    assert is_numeric_dtype(tables_info['size_MB'])
    assert is_numeric_dtype(tables_info['num_tables'])
    non_null = tables_info['num_tables'].dropna()
    assert all(float(x).is_integer() for x in non_null)
