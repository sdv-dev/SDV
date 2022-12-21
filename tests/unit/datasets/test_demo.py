import re

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo, get_available_demos


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


def test_get_available_demos_single_table():
    """Test it can get demos for single table."""
    # Run
    tables_info = get_available_demos('single_table')

    # Assert
    expected_table = pd.DataFrame({
        'dataset_name': [
            'KRK_v1.zip', 'adult.zip', 'alarm.zip', 'asia.zip', 'census.zip',
            'census_extended.zip', 'child.zip', 'covtype.zip', 'credit.zip',
            'expedia_hotel_logs.zip', 'grid.zip', 'gridr.zip', 'insurance.zip',
            'intrusion.zip', 'mnist12.zip', 'mnist28.zip', 'news.zip',
            'ring.zip', 'student_placements.zip', 'student_placements_pii.zip'
        ],
        'size_MB': [
            '0.072128', '3.907448', '4.520128', '1.280128', '98.165608',
            '4.9494', '3.200128', '255.645408', '68.353808', '0.200128',
            '0.320128', '0.320128', '3.340128', '162.039016', '81.200128',
            '439.600128', '18.712096', '0.320128', '0.026358', '0.028078'
        ],
        'num_tables': ['1'] * 20
    })
    pd.testing.assert_frame_equal(tables_info, expected_table)


def test_get_available_demos_multi_table():
    """Test it can get demos for multi table."""
    # Run
    tables_info = get_available_demos('multi_table')

    # Assert
    expected_table = pd.DataFrame({
        'dataset_name': [
            'Accidents_v1.zip', 'Atherosclerosis_v1.zip', 'AustralianFootball_v1.zip',
            'Biodegradability_v1.zip', 'Bupa_v1.zip', 'CORA_v1.zip', 'Carcinogenesis_v1.zip',
            'Chess_v1.zip', 'Countries_v1.zip', 'DCG_v1.zip', 'Dunur_v1.zip', 'Elti_v1.zip',
            'FNHK_v1.zip', 'Facebook_v1.zip', 'Hepatitis_std_v1.zip', 'Mesh_v1.zip',
            'Mooney_Family_v1.zip', 'MuskSmall_v1.zip', 'NBA_v1.zip', 'NCAA_v1.zip',
            'PTE_v1.zip', 'Pima_v1.zip', 'PremierLeague_v1.zip', 'Pyrimidine_v1.zip',
            'SAP_v1.zip', 'SAT_v1.zip', 'SalesDB_v1.zip', 'Same_gen_v1.zip',
            'Student_loan_v1.zip', 'Telstra_v1.zip', 'Toxicology_v1.zip', 'Triazine_v1.zip',
            'TubePricing_v1.zip', 'UTube_v1.zip', 'UW_std_v1.zip', 'WebKP_v1.zip',
            'airbnb-simplified.zip', 'financial_v1.zip', 'ftp_v1.zip', 'genes_v1.zip',
            'got_families.zip', 'imdb_MovieLens_v1.zip', 'imdb_ijs_v1.zip', 'imdb_small_v1.zip',
            'legalActs_v1.zip', 'mutagenesis_v1.zip', 'nations_v1.zip', 'restbase_v1.zip',
            'rossmann.zip', 'trains_v1.zip', 'university_v1.zip', 'walmart.zip', 'world_v1.zip'
        ],
        'size_MB': [
            '296.202744', '7.916808', '32.534832', '0.692008', '0.059144', '1.987328', '1.642592',
            '0.403784', '10.52272', '0.321536', '0.020224', '0.054912', '141.560872', '1.481056',
            '0.809472', '0.101856', '0.121784', '0.646752', '0.16632', '29.137896', '1.31464',
            '0.160896', '17.37664', '0.038144', '196.479272', '0.500224', '325.19768', '0.056176',
            '0.180256', '5.503512', '1.495496', '0.156496', '15.414536', '0.135912', '0.0576',
            '1.9718', '293.14392', '94.718016', '5.45568', '0.440016', '0.001', '55.253264',
            '259.140656', '0.205728', '186.132944', '0.618088', '0.540336', '1.01452', '73.328504',
            '0.00644', '0.009632', '14.642184', '0.295032'
        ],
        'num_tables': [
            '3', '4', '4', '5', '9', '3', '6', '2', '4', '2', '17', '11', '3', '2',
            '7', '29', '68', '2', '4', '9', '38', '9', '4', '2', '4', '36', '4',
            '4', '10', '5', '4', '2', '20', '2', '4', '3', '2', '8', '2', '3', '3',
            '7', '7', '7', '5', '3', '3', '3', '2', '2', '5', '3', '3'
        ]
    })
    pd.testing.assert_frame_equal(tables_info, expected_table)
