import pandas as pd

from sdv.demo import load_demo, load_tabular_demo


def test_load_tabular_demo_default():
    """Test that the default dataset can be accessed in 3 different ways."""
    default_dataset = load_tabular_demo()
    none_dataset = load_tabular_demo(dataset_name=None)
    demo_dataset = load_tabular_demo(dataset_name='demo_single_table')

    expected_columns = [
        'company',
        'department',
        'employee_id',
        'age',
        'age_when_joined',
        'years_in_the_company',
        'salary',
        'annual_bonus',
        'prior_years_experience',
        'full_time',
        'part_time',
        'contractor'
    ]
    expected_company = pd.Series(
        ['Pear', 'Pear', 'Glasses', 'Glasses', 'Cheerper', 'Cheerper'] * 2,
        name='company'
    )
    expected_department = pd.Series(
        ['Sales', 'Design', 'AI', 'Search Engine', 'BigData', 'Support'] * 2,
        name='department'
    )
    expected_employee_id = pd.Series(
        [1, 5, 1, 7, 6, 11, 28, 75, 33, 56, 42, 80],
        name='employee_id'
    )
    expected_full_time = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2, name='full_time')
    expected_part_time = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 2, name='part_time')

    assert expected_columns == list(default_dataset.columns)
    assert expected_columns == list(none_dataset.columns)
    assert expected_columns == list(demo_dataset.columns)
    for dataset in [default_dataset, demo_dataset, none_dataset]:
        pd.testing.assert_series_equal(dataset['company'], expected_company)
        pd.testing.assert_series_equal(dataset['department'], expected_department)
        pd.testing.assert_series_equal(dataset['employee_id'], expected_employee_id)
        pd.testing.assert_series_equal(dataset['full_time'], expected_full_time)
        pd.testing.assert_series_equal(dataset['part_time'], expected_part_time)


def test_load_demo_default():
    """Test that the default datasets can be accessed in 3 different ways."""
    default_datasets = load_demo()
    none_datasets = load_demo(dataset_name=None)
    demo_datasets = load_demo(dataset_name='demo_multi_table')

    for table_name in default_datasets:
        pd.testing.assert_frame_equal(default_datasets[table_name], demo_datasets[table_name])
        pd.testing.assert_frame_equal(default_datasets[table_name], none_datasets[table_name])
