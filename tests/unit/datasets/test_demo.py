import re

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo


def test_download_demo_invalid_modality():
    """Test it crashes when an invalid modality is passed."""
    # Run and Assert
    err_msg = re.escape("'modality' must be in ['single_table', 'multi_table', 'sequential'].")
    with pytest.raises(ValueError, match=err_msg):
        download_demo('invalid_modality', 'dataset_name')


def test_download_demo_folder_already_exists(tmpdir):
    """Test it crashes when folder ``output_folder_name`` already exist."""
    # Run and Assert
    err_msg = re.escape(
        f"Folder '{tmpdir}' already exists. Please specify a different name "
        "or use 'load_csvs' to load from an existing folder."
    )
    with pytest.raises(ValueError, match=err_msg):
        download_demo('single_table', 'dataset_name', tmpdir)


def test_download_demo_dataset_doesnt_exist():
    """Test it crashes when ``dataset_name`` doesn't exist."""
    # Run and Assert
    err_msg = re.escape(
        "Invalid dataset name 'invalid_dataset'. "
        'Make sure you have the correct modality for the dataset name or '
        "use 'get_available_demos' to get a list of demo datasets."
    )
    with pytest.raises(ValueError, match=err_msg):
        download_demo('single_table', 'invalid_dataset')


def test_download_demo_single_table(tmpdir):
    """Test it can download a single table dataset."""
    # Run
    table, metadata = download_demo('single_table', 'ring', tmpdir / 'test_folder')

    # Assert
    expected_table = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    pd.testing.assert_frame_equal(table.head(2), expected_table)

    expected_metadata_dict = {
        'columns': {
            '0': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            },
            '1': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            }
        },
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
    }
    assert metadata.to_dict() == expected_metadata_dict


def test_download_demo_single_table_no_output_folder():
    """Test it can download a single table dataset when no output folder is passed."""
    # Run
    table, metadata = download_demo('single_table', 'ring')

    # Assert
    expected_table = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    pd.testing.assert_frame_equal(table.head(2), expected_table)

    expected_metadata_dict = {
        'columns': {
            '0': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            },
            '1': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            }
        },
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
    }
    assert metadata.to_dict() == expected_metadata_dict


def test_download_demo_timeseries(tmpdir):
    """Test it can download a timeseries dataset."""
    # Run
    table, metadata = download_demo('sequential', 'Libras', tmpdir / 'test_folder')

    # Assert
    expected_table = pd.DataFrame({
        'ml_class': [1, 1],
        'e_id': [0, 0],
        's_index': [0, 1],
        'tt_split': [1, 1],
        'dim_0': [0.67892, 0.68085],
        'dim_1': [0.27315, 0.27315]
    })
    pd.testing.assert_frame_equal(table.head(2), expected_table)

    expected_metadata_dict = {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'e_id': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            },
            'dim_0': {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            },
            'dim_1': {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            },
            'ml_class': {
                'sdtype': 'categorical'
            }
        }
    }
    assert metadata.to_dict() == expected_metadata_dict


def test_download_demo_multi_table(tmpdir):
    """Test it can download a multi table dataset."""
    # Run
    tables, metadata = download_demo('multi_table', 'got_families', tmpdir / 'test_folder')

    # Assert
    expected_families = pd.DataFrame({
        'family_id': [1, 2],
        'name': ['Stark', 'Tully'],
    })
    pd.testing.assert_frame_equal(tables['families.csv'].head(2), expected_families)

    expected_character_families = pd.DataFrame({
        'character_id': [1, 1],
        'family_id': [1, 4],
        'generation': [8, 5],
        'type': ['father', 'mother']
    })
    pd.testing.assert_frame_equal(
        tables['character_families.csv'].head(2), expected_character_families)

    expected_characters = pd.DataFrame({
        'age': [20, 16],
        'character_id': [1, 2],
        'name': ['Jon', 'Arya']
    })
    pd.testing.assert_frame_equal(tables['characters.csv'].head(2), expected_characters)

    expected_metadata_dict = {
        'tables': {
            'characters': {
                'columns': {
                    'character_id': {
                        'sdtype': 'text',
                        'regex_format': '^[1-9]{1,2}$'
                    },
                    'name': {
                        'sdtype': 'categorical'
                    },
                    'age': {
                        'sdtype': 'numerical',
                        'computer_representation': 'Int64'
                    }
                },
                'primary_key': 'character_id'
            },
            'families': {
                'columns': {
                    'family_id': {
                        'sdtype': 'text',
                        'regex_format': '^[1-9]$'
                    },
                    'name': {
                        'sdtype': 'categorical'
                    }
                },
                'primary_key': 'family_id'
            },
            'character_families': {
                'columns': {
                    'character_id': {
                        'sdtype': 'text',
                        'regex_format': '[A-Za-z]{5}'
                    },
                    'family_id': {
                        'sdtype': 'text',
                        'regex_format': '[A-Za-z]{5}'
                    },
                    'type': {
                        'sdtype': 'categorical'
                    },
                    'generation': {
                        'sdtype': 'numerical',
                        'computer_representation': 'Int64'
                    },
                },
                'alternate_keys': ['character_id', 'family_id']
            },
        },
        'relationships': [
            {
                'parent_table_name': 'characters',
                'parent_primary_key': 'character_id',
                'child_table_name': 'character_families',
                'child_foreign_key': 'character_id'
            },
            {
                'parent_table_name': 'families',
                'parent_primary_key': 'family_id',
                'child_table_name': 'character_families',
                'child_foreign_key': 'family_id'
            },
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }
    assert metadata.to_dict() == expected_metadata_dict
