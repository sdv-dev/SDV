import json
import re
from copy import deepcopy
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata._single_table import _SingleTableMetadata
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import (
    compare_metadata,
    compare_ranges,
    download_test_demo,
    get_multi_table_metadata,
)

DEFAULT_TABLE_NAME = 'table'


def test_metadata_to_dict():
    """Test ``to_dict`` method on ``Metadata``."""
    # Setup
    instance = Metadata()

    # Run
    result = instance.to_dict()

    # Assert
    assert result == {'tables': {}, 'relationships': [], 'METADATA_SPEC_VERSION': 'V2'}
    assert instance.tables == {}
    assert instance.relationships == []


def test_load_from_json_single_table_metadata(tmp_path):
    """Test the ``load_from_json`` method with a single table metadata."""
    # Setup
    old_metadata = _SingleTableMetadata.load_from_dict({
        'columns': {
            'column_1': {'sdtype': 'numerical'},
            'column_2': {'sdtype': 'categorical'},
        },
    })
    old_metadata.save_to_json(tmp_path / 'metadata.json')
    expected_warning = re.escape(
        'You are loading an older _SingleTableMetadata object. This will be converted '
        f"into the new Metadata object with a placeholder table name ('{DEFAULT_TABLE_NAME}')."
        ' Please save this new object for future usage.'
    )

    # Run
    with pytest.warns(UserWarning, match=expected_warning):
        metadata = Metadata.load_from_json(tmp_path / 'metadata.json')

    # Assert
    assert metadata.to_dict() == {
        'tables': {
            DEFAULT_TABLE_NAME: {
                'columns': {
                    'column_1': {'sdtype': 'numerical'},
                    'column_2': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }


def test_detect_from_dataframes_multi_table():
    """Test the ``detect_from_dataframes`` method works with multi-table."""
    # Setup
    real_data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data)

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'categorical'},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    compare_metadata(metadata, expected_metadata)
    compare_ranges(metadata, real_data)


def test_detect_from_dataframes_multi_table_without_infer_sdtypes():
    """Test it when infer_sdtypes is False."""
    # Setup
    real_data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_sdtypes=False)

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'unknown', 'pii': True},
                    'state': {'sdtype': 'unknown', 'pii': True},
                    'rating': {'sdtype': 'unknown', 'pii': True},
                    'classification': {'sdtype': 'unknown', 'pii': True},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'id'},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'unknown', 'pii': True},
                    'room_type': {'sdtype': 'unknown', 'pii': True},
                    'amenities_fee': {'sdtype': 'unknown', 'pii': True},
                    'checkin_date': {'sdtype': 'unknown', 'pii': True},
                    'checkout_date': {'sdtype': 'unknown', 'pii': True},
                    'room_rate': {'sdtype': 'unknown', 'pii': True},
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
                    'credit_card_number': {'sdtype': 'unknown', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'child_foreign_key': 'hotel_id',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'parent_table_name': 'hotels',
            }
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    compare_metadata(metadata, expected_metadata)
    compare_ranges(metadata, real_data)


def test_detect_from_dataframes_multi_table_with_infer_keys_primary_only():
    """Test it when infer_keys is 'primary_only'."""
    # Setup
    real_data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_keys='primary_only')

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'categorical'},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }
    compare_metadata(metadata, expected_metadata)
    compare_ranges(metadata, real_data)


def test_detect_from_dataframes_multi_table_with_infer_keys_none():
    """Test it when infer_keys is None."""
    # Setup
    real_data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_keys=None)

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'categorical'},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }
    compare_metadata(metadata, expected_metadata)
    compare_ranges(metadata, real_data)


def test_detect_from_dataframes_single_table():
    """Test the ``detect_from_dataframes`` method works with a single table."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table_1': data['hotels']}
    metadata = Metadata.detect_from_dataframes(data)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            'table_1': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframes_single_table_infer_sdtypes_false():
    """Test it for a single table when infer_sdtypes is False."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table_1': data['hotels']}
    metadata = Metadata.detect_from_dataframes(data, infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            'table_1': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'unknown', 'pii': True},
                    'state': {'sdtype': 'unknown', 'pii': True},
                    'rating': {'sdtype': 'unknown', 'pii': True},
                    'classification': {'sdtype': 'unknown', 'pii': True},
                },
                'primary_key': 'hotel_id',
            }
        },
        'relationships': [],
    }
    compare_metadata(metadata, expected_metadata)
    compare_ranges(metadata, data)


def test_detect_from_dataframes_single_table_infer_keys_primary_only():
    """Test it for a single table when infer_keys is 'primary_only'."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table_1': data['hotels']}
    metadata = Metadata.detect_from_dataframes(data, infer_keys='primary_only')

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            'table_1': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframes_single_table_infer_keys_none():
    """Test it for a single table when infer_keys is None."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table_1': data['hotels']}
    metadata = Metadata.detect_from_dataframes(data, infer_keys=None)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            'table_1': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframe():
    """Test that a single table can be detected as a DataFrame."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table': data['hotels']}

    metadata = Metadata.detect_from_dataframe(data, 'table')

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            DEFAULT_TABLE_NAME: {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframe_infer_sdtypes_false():
    """Test it when infer_sdtypes is False."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table': data['hotels']}
    metadata = Metadata.detect_from_dataframe(data, 'table', infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            DEFAULT_TABLE_NAME: {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'unknown', 'pii': True},
                    'state': {'sdtype': 'unknown', 'pii': True},
                    'rating': {'sdtype': 'unknown', 'pii': True},
                    'classification': {'sdtype': 'unknown', 'pii': True},
                },
                'primary_key': 'hotel_id',
            },
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframe_infer_keys_none():
    """Test it when infer_keys is None."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table': data['hotels']}
    metadata = Metadata.detect_from_dataframe(data, 'table', infer_keys=None)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            DEFAULT_TABLE_NAME: {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_dataframe_infer_keys_none_infer_sdtypes_false():
    """Test it when infer_keys is None and infer_sdtypes is False."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    data = {'table': data['hotels']}
    metadata = Metadata.detect_from_dataframe(data, 'table', infer_keys=None, infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'tables': {
            DEFAULT_TABLE_NAME: {
                'columns': {
                    'hotel_id': {'sdtype': 'unknown', 'pii': True},
                    'city': {'sdtype': 'unknown', 'pii': True},
                    'state': {'sdtype': 'unknown', 'pii': True},
                    'rating': {'sdtype': 'unknown', 'pii': True},
                    'classification': {'sdtype': 'unknown', 'pii': True},
                },
            }
        },
        'relationships': [],
    }
    compare_ranges(metadata, data)
    compare_metadata(metadata, expected_metadata)


def test_detect_from_csvs(tmp_path):
    """Test the ``detect_from_csvs`` method."""
    # Setup
    real_data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_from_csvs(folder_name=tmp_path)

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'categorical'},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }

    compare_ranges(metadata, real_data)
    compare_metadata(metadata, expected_metadata)


params = [
    ('update_column', ['column_name'], {'column_name': 'has_rewards', 'sdtype': 'categorical'}),
    (
        'update_columns',
        ['column_names'],
        {'column_names': ['has_rewards', 'billing_address'], 'sdtype': 'categorical'},
    ),
    (
        'update_columns_metadata',
        ['column_metadata'],
        {'column_metadata': {'has_rewards': {'sdtype': 'categorical'}}},
    ),
    ('add_column', ['column_name'], {'column_name': 'has_rewards_2', 'sdtype': 'categorical'}),
    ('set_primary_key', ['column_name'], {'column_name': 'billing_address'}),
    ('remove_primary_key', [], {}),
    (
        'add_column_relationship',
        ['relationship_type', 'column_names'],
        {'column_names': ['billing_address'], 'relationship_type': 'address'},
    ),
    ('add_alternate_keys', ['column_names'], {'column_names': ['billing_address']}),
    ('set_sequence_key', ['column_name'], {'column_name': 'billing_address'}),
    ('get_column_names', [], {'sdtype': 'datetime'}),
]


@pytest.mark.parametrize('method, args, kwargs', params)
def test_any_metadata_update_single_table(method, args, kwargs):
    """Test that any method that updates metadata works for single-table case."""
    # Setup
    _, metadata = download_test_demo('single_table', 'fake_hotel_guests')
    metadata.update_column(
        table_name='fake_hotel_guests', column_name='billing_address', sdtype='street_address'
    )
    parameter = [kwargs[arg] for arg in args]
    remaining_kwargs = {key: value for key, value in kwargs.items() if key not in args}
    metadata_before = deepcopy(metadata).to_dict()

    # Run
    result = getattr(metadata, method)(*parameter, **remaining_kwargs)

    # Assert
    expected_dict = metadata.to_dict()
    if method != 'get_column_names':
        assert expected_dict != metadata_before
    else:
        assert result == ['checkin_date', 'checkout_date']


@pytest.mark.parametrize('method, args, kwargs', params)
def test_any_metadata_update_multi_table(method, args, kwargs):
    """Test that any method that updates metadata works for multi-table case."""
    # Setup
    _, metadata = download_test_demo('multi_table', 'fake_hotels')
    metadata.update_column(
        table_name='guests', column_name='billing_address', sdtype='street_address'
    )
    parameter = [kwargs[arg] for arg in args]
    remaining_kwargs = {key: value for key, value in kwargs.items() if key not in args}
    metadata_before = deepcopy(metadata).to_dict()
    expected_error = re.escape(
        'Metadata contains more than one table, please specify the `table_name`.'
    )

    # Run
    with pytest.raises(ValueError, match=expected_error):
        getattr(metadata, method)(*parameter, **remaining_kwargs)

    parameter.append('guests')
    result = getattr(metadata, method)(*parameter, **remaining_kwargs)

    # Assert
    expected_dict = metadata.to_dict()
    if method != 'get_column_names':
        assert expected_dict != metadata_before
    else:
        assert result == ['checkin_date', 'checkout_date']


def test_anonymize():
    """Test the ``anonymize`` method."""
    # Setup
    metadata_dict = {
        'tables': {
            'real_table1': {
                'columns': {
                    'table1_primary_key': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{3}'},
                    'table1_column2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table1_primary_key',
            },
            'real_table2': {
                'columns': {
                    'table2_primary_key': {'sdtype': 'email'},
                    'table2_foreign_key': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{3}'},
                },
                'primary_key': 'table2_primary_key',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'real_table1',
                'parent_primary_key': 'table1_primary_key',
                'child_table_name': 'real_table2',
                'child_foreign_key': 'table2_foreign_key',
            }
        ],
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    table1_metadata = metadata.tables['real_table1']
    table2_metadata = metadata.tables['real_table2']
    metadata.validate()

    # Run
    anonymized = metadata.anonymize()

    # Assert
    anonymized.validate()

    assert anonymized.METADATA_SPEC_VERSION == 'V2'
    assert anonymized.tables.keys() == {'table1', 'table2'}
    assert len(anonymized.relationships) == len(metadata.relationships)
    assert anonymized.relationships[0]['parent_table_name'] == 'table1'
    assert anonymized.relationships[0]['child_table_name'] == 'table2'
    assert anonymized.relationships[0]['parent_primary_key'] == 'col1'
    assert anonymized.relationships[0]['child_foreign_key'] == 'col2'

    anon_primary_key_metadata = anonymized.tables['table1'].columns['col1']
    assert anon_primary_key_metadata == table1_metadata.columns['table1_primary_key']

    anon_foreign_key_metadata = anonymized.tables['table2'].columns['col2']
    assert anon_foreign_key_metadata == table2_metadata.columns['table2_foreign_key']

    assert anonymized.tables['table1'].to_dict() == table1_metadata.anonymize().to_dict()
    assert anonymized.tables['table2'].to_dict() == table2_metadata.anonymize().to_dict()


def test_detect_from_dataframes_invalid_format():
    """Test the ``detect_from_dataframes`` method with an invalid data format."""
    # Setup
    dict_data = [
        {
            'key1': i,
            'key2': f'string_{i}',
            'key3': 1.5,
        }
        for i in range(100)
    ]
    data = {
        'table_1': pd.DataFrame({
            'dict_column': dict_data,
            'numerical': [1.2] * 100,
        }),
        'table_2': pd.DataFrame({
            'numerical': [1.5] * 10,
            'categorical': ['A'] * 10,
        }),
    }
    expected_error = re.escape(
        "Unable to detect metadata for table 'table_1' column 'dict_column'. This may be because "
        "the data type is not supported.\n TypeError: unhashable type: 'dict'"
    )

    # Run and Assert
    with pytest.raises(InvalidMetadataError, match=expected_error):
        Metadata.detect_from_dataframes(data)


def test_detect_from_dataframes__primary_to_primary():
    """Test metadata auto-detection works for primary to primary relationships."""
    # Setup
    data = {
        'tableA': pd.DataFrame({
            'table_A_id': range(5),
            'column_1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_A_id': range(5),
            'column_2': ['A', 'B', 'B', 'C', 'C'],
        }),
    }

    # Run
    detected_metadata = Metadata().detect_from_dataframes(
        data, foreign_key_inference_algorithm='column_name_match'
    )

    # Assert
    assert detected_metadata.tables['tableA'].primary_key == 'table_A_id'
    assert detected_metadata.tables['tableB'].primary_key == 'table_A_id'
    assert detected_metadata.relationships == [
        {
            'parent_table_name': 'tableA',
            'child_table_name': 'tableB',
            'parent_primary_key': 'table_A_id',
            'child_foreign_key': 'table_A_id',
        }
    ]


def test_detect_from_dataframes__primary_to_primary_no_cycles():
    """Test metadata auto-detection does not create cycles with PK to PK."""
    # Setup
    data = {
        'tableA': pd.DataFrame({
            'table_A_id': range(5),
            'column_1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_A_id': range(5),
            'column_2': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableC': pd.DataFrame({
            'table_A_id': range(5),
            'column_2': ['A', 'B', 'B', 'C', 'C'],
        }),
    }

    # Run
    detected_metadata = Metadata().detect_from_dataframes(
        data, foreign_key_inference_algorithm='column_name_match'
    )

    # Assert
    assert detected_metadata.tables['tableA'].primary_key == 'table_A_id'
    assert detected_metadata.tables['tableB'].primary_key == 'table_A_id'
    assert detected_metadata.tables['tableC'].primary_key == 'table_A_id'
    assert len(detected_metadata.relationships) == 2
    assert {
        'parent_table_name': 'tableA',  # PK to PK
        'child_table_name': 'tableC',
        'parent_primary_key': 'table_A_id',
        'child_foreign_key': 'table_A_id',
    } in detected_metadata.relationships
    assert {
        'parent_table_name': 'tableA',  # PK to PK
        'child_table_name': 'tableB',
        'parent_primary_key': 'table_A_id',
        'child_foreign_key': 'table_A_id',
    } in detected_metadata.relationships


def test_validate_metadata_with_reused_foreign_keys():
    # Setup
    metadata_dict = {
        'tables': {
            'A1': {
                'columns': {
                    'data': {'sdtype': 'numerical'},
                    'id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                },
                'primary_key': 'id',
            },
            'A2': {
                'columns': {
                    'data': {'sdtype': 'numerical'},
                    'id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'fk1_A1': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                },
                'primary_key': 'id',
            },
            'A3': {
                'columns': {
                    'data': {'sdtype': 'numerical'},
                    'id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'fk1_A1': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'fk2_A1': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'fk3_A1_A2': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                },
                'primary_key': 'id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'A1',
                'parent_primary_key': 'id',
                'child_table_name': 'A2',
                'child_foreign_key': 'fk1_A1',
            },
            {
                'parent_table_name': 'A1',
                'parent_primary_key': 'id',
                'child_table_name': 'A3',
                'child_foreign_key': 'fk1_A1',
            },
            {
                'parent_table_name': 'A1',
                'parent_primary_key': 'id',
                'child_table_name': 'A3',
                'child_foreign_key': 'fk2_A1',
            },
            {
                'parent_table_name': 'A1',
                'parent_primary_key': 'id',
                'child_table_name': 'A3',
                'child_foreign_key': 'fk3_A1_A2',
            },
            {
                'parent_table_name': 'A2',
                'parent_primary_key': 'id',
                'child_table_name': 'A3',
                'child_foreign_key': 'fk3_A1_A2',
            },
            {
                'parent_table_name': 'A3',
                'parent_primary_key': 'id',
                'child_table_name': 'A4',
                'child_foreign_key': 'fk1_A3',
            },
        ],
    }

    metadata = Metadata.load_from_dict(metadata_dict)
    # Run and Assert
    error_msg = re.escape(
        'Relationships:\n'
        'Relationship between tables (A2, A3) uses a foreign key '
        "('fk3_A1_A2') that is already used in another relationship."
    )
    with pytest.raises(InvalidMetadataError, match=error_msg):
        metadata.validate()


@pytest.fixture
def metadata_instance():
    metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'gender': {'sdtype': 'categorical'},
                    'age': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                    'user_id': {'sdtype': 'id'},
                },
                'primary_key': 'user_id',
            },
            'transactions': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'transaction_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'numerical'},
                },
                'primary_key': 'transaction_id',
            },
            'products': {
                'columns': {
                    'product_id': {'sdtype': 'id'},
                    'cost': {'sdtype': 'numerical'},
                    'weight': {'sdtype': 'numerical'},
                    'manufacturer': {'sdtype': 'id'},
                },
                'primary_key': 'product_id',
            },
            'manufacturers': {
                'columns': {
                    'country': {'sdtype': 'categorical'},
                    'address': {'sdtype': 'text'},
                    'id': {'sdtype': 'id'},
                },
                'column_relationships': [
                    {
                        'type': 'address',
                        'column_names': [
                            'country',
                            'address',
                        ],
                    },
                ],
                'primary_key': 'id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'products',
                'parent_primary_key': 'product_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'product_id',
            },
            {
                'parent_table_name': 'manufacturers',
                'parent_primary_key': 'id',
                'child_table_name': 'products',
                'child_foreign_key': 'manufacturer',
            },
        ],
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    return metadata


def test_remove_table(metadata_instance):
    """Test that a table and all relationships it has are removed."""
    # Run
    metadata_instance.remove_table('products')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'gender': {'sdtype': 'categorical'},
                    'age': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                    'user_id': {'sdtype': 'id'},
                },
                'primary_key': 'user_id',
            },
            'transactions': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'transaction_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'numerical'},
                },
                'primary_key': 'transaction_id',
            },
            'manufacturers': {
                'columns': {
                    'country': {'sdtype': 'categorical'},
                    'address': {'sdtype': 'text'},
                    'id': {'sdtype': 'id'},
                },
                'column_relationships': [
                    {
                        'type': 'address',
                        'column_names': [
                            'country',
                            'address',
                        ],
                    },
                ],
                'primary_key': 'id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'user_id',
            },
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == metadata_instance.to_dict()
    assert metadata_instance._multi_table_updated


def test_remove_column(metadata_instance):
    """Test that the column is removed from all relationships and keys."""
    # Run
    metadata_instance.remove_column('id', 'manufacturers')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'gender': {'sdtype': 'categorical'},
                    'age': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                    'user_id': {'sdtype': 'id'},
                },
                'primary_key': 'user_id',
            },
            'transactions': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'transaction_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'numerical'},
                },
                'primary_key': 'transaction_id',
            },
            'products': {
                'columns': {
                    'product_id': {'sdtype': 'id'},
                    'cost': {'sdtype': 'numerical'},
                    'weight': {'sdtype': 'numerical'},
                    'manufacturer': {'sdtype': 'id'},
                },
                'primary_key': 'product_id',
            },
            'manufacturers': {
                'columns': {
                    'country': {'sdtype': 'categorical'},
                    'address': {'sdtype': 'text'},
                },
                'column_relationships': [
                    {
                        'type': 'address',
                        'column_names': [
                            'country',
                            'address',
                        ],
                    },
                ],
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'products',
                'parent_primary_key': 'product_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'product_id',
            },
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == metadata_instance.to_dict()
    assert metadata_instance._multi_table_updated


def test_remove_column_column_relationships(metadata_instance):
    """Test that all column relationships the column is in are removed."""
    # Run
    metadata_instance.remove_column('address', 'manufacturers')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'gender': {'sdtype': 'categorical'},
                    'age': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                    'user_id': {'sdtype': 'id'},
                },
                'primary_key': 'user_id',
            },
            'transactions': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'transaction_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'numerical'},
                },
                'primary_key': 'transaction_id',
            },
            'products': {
                'columns': {
                    'product_id': {'sdtype': 'id'},
                    'cost': {'sdtype': 'numerical'},
                    'weight': {'sdtype': 'numerical'},
                    'manufacturer': {'sdtype': 'id'},
                },
                'primary_key': 'product_id',
            },
            'manufacturers': {
                'columns': {
                    'country': {'sdtype': 'categorical'},
                    'id': {'sdtype': 'id'},
                },
                'primary_key': 'id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'user_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'products',
                'parent_primary_key': 'product_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'product_id',
            },
            {
                'parent_table_name': 'manufacturers',
                'parent_primary_key': 'id',
                'child_table_name': 'products',
                'child_foreign_key': 'manufacturer',
            },
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == metadata_instance.to_dict()
    assert metadata_instance._multi_table_updated


@pytest.fixture
def sequential_metadata():
    metadata_dict = {
        'tables': {
            'trades': {
                'columns': {
                    'ticker': {'sdtype': 'id'},
                    'cost': {'sdtype': 'numerical'},
                    'quantity': {'sdtype': 'numerical'},
                    'time': {'sdtype': 'datetime'},
                },
                'primary_key': 'transaction_id',
                'sequence_key': 'ticker',
                'sequence_index': 'time',
            },
        }
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    return metadata


def test_remove_column_column_is_sequence_key(sequential_metadata):
    """Test that a column is properly removed if it is a sequence key."""
    # Run
    sequential_metadata.remove_column('ticker')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'trades': {
                'columns': {
                    'cost': {'sdtype': 'numerical'},
                    'quantity': {'sdtype': 'numerical'},
                    'time': {'sdtype': 'datetime'},
                },
                'primary_key': 'transaction_id',
                'sequence_index': 'time',
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == sequential_metadata.to_dict()
    assert sequential_metadata._multi_table_updated


def test_remove_column_column_is_sequence_index(sequential_metadata):
    """Test that a column is properly removed if it is a sequence index."""
    # Run
    sequential_metadata.remove_column('time')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'trades': {
                'columns': {
                    'ticker': {'sdtype': 'id'},
                    'cost': {'sdtype': 'numerical'},
                    'quantity': {'sdtype': 'numerical'},
                },
                'primary_key': 'transaction_id',
                'sequence_key': 'ticker',
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == sequential_metadata.to_dict()
    assert sequential_metadata._multi_table_updated


def test_remove_column_alternate_key():
    """Test that the column is removed from the alternate keys if it is one."""
    # Setup
    metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'email': {'sdtype': 'id'},
                    'id': {'sdtype': 'id'},
                    'age': {'sdtype': 'numerical'},
                },
                'primary_key': 'id',
                'alternate_keys': ['email'],
            },
        }
    }
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    metadata.remove_column('email')

    # Assert
    expected_metadata_dict = {
        'tables': {
            'users': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'age': {'sdtype': 'numerical'},
                },
                'primary_key': 'id',
            },
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert expected_metadata_dict == metadata.to_dict()
    assert metadata._multi_table_updated


def test_loading_invalid_single_table_metadata():
    """Test loading invalid single table metadata dict."""
    # Setup
    _, metadata = download_test_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata_dict = metadata.to_dict()
    metadata_dict['tables']['guests']['invalid_key'] = {'value1': True, 'value2': False}
    expected_error = re.escape(
        "Invalid metadata dict for table 'guests':\n "
        "The metadata dictionary contains extra keys: 'invalid_key'. "
        "Valid keys are: 'METADATA_SPEC_VERSION', 'alternate_keys', "
        "'column_relationships', 'columns', 'primary_key', 'sequence_index',"
        " 'sequence_key'."
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        Metadata.load_from_dict(metadata_dict)


def test_validate_empty_metadata():
    """Test that the metadata is invalid if it is empty."""
    # Setup
    metadata = Metadata.load_from_dict({})

    # Run and Assert
    err_msg = re.escape(
        "The metadata is not valid\nTable 'table' has 0 columns. "
        "Use 'add_column' to specify its columns."
    )
    with pytest.raises(InvalidMetadataError, match=err_msg):
        GaussianCopulaSynthesizer(metadata)


def test_validate_pk_to_pk(primary_key_to_primary_key):
    """Test validation to indicate a PK to PK relationship."""
    # Setup
    data, metadata_instance = primary_key_to_primary_key

    # Run and Assert
    metadata_instance.validate()
    metadata_instance.validate_data(data)


def test_validate_pk_to_pk_email():
    """Test validation with PK to PK and email sdtype."""
    # Setup
    metadata_instance = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'columns': {
                    'table_A_primary_key': {'sdtype': 'email'},
                    'column_1': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_A_primary_key',
            },
            'tableB': {
                'columns': {
                    'table_B_primary_key': {'sdtype': 'email'},
                    'column_2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_B_primary_key',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'tableA',
                'parent_primary_key': 'table_A_primary_key',
                'child_table_name': 'tableB',
                'child_foreign_key': 'table_B_primary_key',
            }
        ],
    })
    data = {
        'tableA': pd.DataFrame({
            'table_A_primary_key': [
                'user1@domain.com',
                'user2@domain.com',
                'user3@domain.com',
                'user4@domain.com',
                'user5@domain.com',
            ],
            'column_1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_B_primary_key': [
                'user1@domain.com',
                'user2@domain.com',
                'user3@domain.com',
                'user4@domain.com',
                'user5@domain.com',
            ],
            'column_2': ['A', 'B', 'B', 'C', 'C'],
        }),
    }

    # Run and Assert
    metadata_instance.validate()
    metadata_instance.validate_data(data)


def test_set_primary_key_pk_to_pk():
    """Test set_primary_key to indicate a PK to PK relationship."""
    # Setup
    metadata_instance = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'columns': {
                    'table_A_primary_key': {'sdtype': 'id'},
                    'column_1': {'sdtype': 'categorical'},
                }
            },
            'tableB': {
                'columns': {
                    'table_B_primary_key': {'sdtype': 'id'},
                    'column_2': {'sdtype': 'categorical'},
                }
            },
        },
        'relationships': [],
    })

    # Run
    metadata_instance.set_primary_key(
        table_name='tableA',
        column_name='table_A_primary_key',
    )
    metadata_instance.set_primary_key(
        table_name='tableB',
        column_name='table_B_primary_key',
    )
    metadata_instance.relationships = [
        {
            'parent_table_name': 'tableB',
            'parent_primary_key': 'table_B_primary_key',
            'child_table_name': 'tableA',
            'child_foreign_key': 'table_A_primary_key',
        }
    ]

    # Assert
    expected_metadata = {
        'tables': {
            'tableA': {
                'columns': {
                    'table_A_primary_key': {'sdtype': 'id'},
                    'column_1': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_A_primary_key',
            },
            'tableB': {
                'columns': {
                    'table_B_primary_key': {'sdtype': 'id'},
                    'column_2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_B_primary_key',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'tableB',
                'parent_primary_key': 'table_B_primary_key',
                'child_table_name': 'tableA',
                'child_foreign_key': 'table_A_primary_key',
            }
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert metadata_instance.to_dict() == expected_metadata


@pytest.mark.parametrize(
    'parent_table_name, child_table_name, parent_primary_key, child_foreign_key',
    [
        ('tableA', 'tableB', 'table_A_primary_key', 'table_B_primary_key'),
        ('tableB', 'tableA', 'table_B_primary_key', 'table_A_primary_key'),
    ],
)
def test_add_relationship_pk_to_pk(
    parent_table_name, child_table_name, parent_primary_key, child_foreign_key
):
    """Test add a relationship to indicate a PK to PK relationship."""
    # Setup
    metadata_instance = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'primary_key': 'table_A_primary_key',
                'columns': {
                    'table_A_primary_key': {'sdtype': 'id'},
                    'column_1': {'sdtype': 'categorical'},
                },
            },
            'tableB': {
                'primary_key': 'table_B_primary_key',
                'columns': {
                    'table_B_primary_key': {'sdtype': 'id'},
                    'column_2': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [],
    })

    # Run
    metadata_instance.add_relationship(
        parent_table_name=parent_table_name,
        child_table_name=child_table_name,
        parent_primary_key=parent_primary_key,
        child_foreign_key=child_foreign_key,
    )

    # Assert
    assert metadata_instance.to_dict()['tables'] == {
        'tableA': {
            'columns': {
                'table_A_primary_key': {'sdtype': 'id'},
                'column_1': {'sdtype': 'categorical'},
            },
            'primary_key': 'table_A_primary_key',
        },
        'tableB': {
            'columns': {
                'table_B_primary_key': {'sdtype': 'id'},
                'column_2': {'sdtype': 'categorical'},
            },
            'primary_key': 'table_B_primary_key',
        },
    }
    assert metadata_instance.to_dict()['relationships'] == [
        {
            'parent_table_name': parent_table_name,
            'child_table_name': child_table_name,
            'parent_primary_key': parent_primary_key,
            'child_foreign_key': child_foreign_key,
        }
    ]


def test_add_column_relationship_fails_with_primary_key_column():
    """Test that adding a column relationship fails if the column is part of the primary key.

    This test also adds a `billing` mutation to the column relationship types
    for `_SingleTableMetadata`. The error that is being raised otherwise
    is `ImportError` instead of `InvalidMetadataError`.
    """
    # Setup
    data, metadata = download_test_demo(modality='single_table', dataset_name='fake_hotel_guests')
    metadata.update_column(column_name='billing_address', sdtype='street_address')
    metadata.set_primary_key(['guest_email', 'billing_address'])
    expected_msg = "Cannot use primary key 'billing_address' in column relationship."
    _SingleTableMetadata._COLUMN_RELATIONSHIP_TYPES['billing'] = Mock()

    # Run and Assert
    with pytest.raises(InvalidMetadataError, match=expected_msg):
        metadata.add_column_relationship(
            column_names=['billing_address'], relationship_type='billing'
        )

    # Test cleanup: remove 'billing' from the class-level relationship types.
    # Without this, the mutation would leak into later _SingleTableMetadata instances.
    _SingleTableMetadata._COLUMN_RELATIONSHIP_TYPES.pop('billing')


def test_metadata_fails_for_relationship_with_set_primary_key_column_in_relationship():
    """Test metadata set_primary_key fails if a column relationship includes primary key column."""
    # Setup
    data, metadata = download_test_demo(modality='single_table', dataset_name='fake_hotel_guests')
    metadata.update_column(column_name='billing_address', sdtype='street_address')
    metadata.add_column_relationship(column_names=['billing_address'], relationship_type='address')
    expected_msg = r"Cannot set primary key '.*' because it is part of a column relationship\."

    # Run and Assert
    with pytest.raises(InvalidMetadataError, match=expected_msg):
        metadata.set_primary_key(['guest_email', 'billing_address'])


def test_metadata_fails_with_proper_message_when_setting_primary_key():
    """Test that when setting a primary key with no id columns it will fail."""
    # Setup
    account_metadata = Metadata.load_from_dict({
        'tables': {
            'accounts': {
                'columns': {
                    'user_id': {'sdtype': 'id', 'regex_format': 'ID_[0-9]{1,2}'},
                    'account_type': {'sdtype': 'categorical'},
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'},
                }
            }
        }
    })

    # Run and Assert
    expected_msg = re.escape(
        "The primary_keys ['col1', 'col2'] must have a column of type 'id' or another PII type."
    )
    with pytest.raises(InvalidMetadataError, match=expected_msg):
        account_metadata.set_primary_key(['col1', 'col2'])


def test_detect_from_dataframe_verbose_single(capsys):
    """Test 'detect_from_dataframe' with verbose True with single table."""
    # Setup
    data, _ = download_test_demo(modality='single_table', dataset_name='fake_hotel_guests')
    data = {'table': data['fake_hotel_guests']}

    # Run
    metadata = Metadata.detect_from_dataframe(data, 'table', verbose=True)

    # Assert
    captured = capsys.readouterr().out
    expected_output = [
        "\nDetecting table 'table':\n",
        "- Column 'guest_email': sdtype='email', pii=True\n",
        "- Column 'has_rewards': sdtype='categorical', range_is_nullable=",
        "- Column 'room_type': sdtype='categorical', range_is_nullable=",
        "- Column 'amenities_fee': sdtype='numerical', range_is_nullable=",
        "- Column 'checkin_date': sdtype='datetime', datetime_format='%d %b %Y', "
        'range_is_nullable=',
        "- Column 'checkout_date': sdtype='datetime', datetime_format='%d %b %Y', "
        'range_is_nullable=',
        "- Column 'room_rate': sdtype='numerical', range_is_nullable=",
        "- Column 'billing_address': sdtype='categorical', range_is_nullable=",
        "- Column 'credit_card_number': sdtype='credit_card_number', pii=True, range_is_nullable=",
        "\nDetecting primary key for table 'table':\n",
        "- primary_key='guest_email'\n",
    ]
    for line in expected_output:
        assert line in captured

    assert list(metadata.tables.keys()) == ['table']
    assert list(metadata.tables['table'].columns.keys()) == [
        'guest_email',
        'has_rewards',
        'room_type',
        'amenities_fee',
        'checkin_date',
        'checkout_date',
        'room_rate',
        'billing_address',
        'credit_card_number',
    ]


def test_detect_from_dataframes_verbose(capsys):
    """Test 'detect_from_dataframe' with verbose True with multi table."""
    # Setup
    data, _ = download_test_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(data, verbose=True)

    # Assert
    captured = capsys.readouterr().out
    expected_output = [
        "\nDetecting table 'guests':\n",
        "- Column 'guest_email': sdtype='email', pii=True\n",
        "- Column 'hotel_id': sdtype='id', range_is_nullable=",
        "- Column 'has_rewards': sdtype='categorical', range_is_nullable=",
        "- Column 'amenities_fee': sdtype='numerical', range_is_nullable=",
        "- Column 'checkin_date': sdtype='datetime', datetime_format='%d %b %Y', "
        'range_is_nullable=',
        "\nDetecting primary key for table 'guests':\n",
        "- primary_key='guest_email'\n",
        "\nDetecting table 'hotels':\n",
        "- Column 'hotel_id': sdtype='id'\n",
        "- Column 'city': sdtype='city', pii=True, range_is_nullable=",
        "- Column 'state': sdtype='administrative_unit', pii=True, range_is_nullable=",
        "- Column 'rating': sdtype='numerical', range_is_nullable=",
        "- Column 'classification': sdtype='categorical', range_is_nullable=",
        "\nDetecting primary key for table 'hotels':\n",
        "- primary_key='hotel_id'\n",
        '\nDetecting foreign keys:\n',
        "- Column 'guests.hotel_id' refers to column 'hotels.hotel_id'\n",
    ]
    for line in expected_output:
        assert line in captured

    assert list(metadata.tables.keys()) == ['guests', 'hotels']


def test_detect_from_dataframes_verbose_updates_fk_sdtype(capsys):
    """Test 'detect_from_dataframes' verbose output when a FK sdtype is updated to 'id'."""
    # Setup
    data = {
        'users': pd.DataFrame({
            'account': [f'acct_{i}' for i in range(10)],
        }),
        'transactions': pd.DataFrame({
            'transaction_id': range(10),
            'account': [f'acct_{i}' for i in range(10)],
        }),
    }
    expected_output = (
        "\nDetecting table 'users':\n"
        "- Column 'account': sdtype='id'\n\n"
        "Detecting primary key for table 'users':\n"
        "- primary_key='account' (updating sdtype to 'id')\n\n"
        "Detecting table 'transactions':\n"
        "- Column 'transaction_id': sdtype='id'\n"
        "- Column 'account': sdtype='categorical', range_is_nullable=False, "
        "range_values=['acct_0', 'acct_1', 'acct_2', 'acct_3', 'acct_4', 'acct_5', "
        "'acct_6', 'acct_7', 'acct_8', 'acct_9']\n\n"
        "Detecting primary key for table 'transactions':\n"
        "- primary_key='transaction_id'\n\n"
        'Detecting foreign keys:\n'
        "- Column 'transactions.account' refers to column "
        "'users.account' (updating sdtype to 'id')\n"
    )

    # Run
    metadata = Metadata.detect_from_dataframes(
        data, foreign_key_inference_algorithm='column_name_match', verbose=True
    )

    # Assert
    captured = capsys.readouterr().out
    assert expected_output == captured
    assert metadata.tables['users'].columns['account']['sdtype'] == 'id'
    assert metadata.tables['transactions'].columns['account']['sdtype'] == 'id'


def test_detect_from_dataframes_small_dataset():
    """Test `detect_from_dataframes` by comparing with the expected metadata."""
    # Setup
    num_rows = 50
    data = {
        'users': pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(num_rows)],
            'age': range(20, 20 + num_rows),
            'signup_date': [str(d) for d in pd.date_range('2026-01-01', periods=num_rows)],
            'is_active': [True, False] * 25,
        }),
        'transactions': pd.DataFrame({
            'transaction_id': [f'transaction_{i}' for i in range(num_rows)],
            'user_id': [f'user_{i}' for i in range(num_rows)],
            'category': ['food', 'travel'] * 25,
            'rating': [1, 2, 3, 4, 5] * 10,
            'amount': [10.5 + i for i in range(num_rows)],
        }),
    }

    expected_metadata = {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {
                        'sdtype': 'id',
                    },
                    'age': {
                        'sdtype': 'numerical',
                        'range_is_nullable': False,
                        'range_min': 20,
                        'range_max': 69,
                        'decimal_places': 0,
                    },
                    'signup_date': {
                        'sdtype': 'datetime',
                        'datetime_format': '%Y-%m-%d %H:%M:%S',
                        'range_is_nullable': False,
                        'range_min': '2026-01-01 00:00:00',
                        'range_max': '2026-02-19 00:00:00',
                    },
                    'is_active': {
                        'sdtype': 'categorical',
                        'range_is_nullable': False,
                        'range_values': [True, False],
                    },
                },
            },
            'transactions': {
                'primary_key': 'transaction_id',
                'columns': {
                    'transaction_id': {
                        'sdtype': 'id',
                    },
                    'user_id': {
                        'sdtype': 'id',
                        'range_is_nullable': False,
                    },
                    'category': {
                        'sdtype': 'categorical',
                        'range_is_nullable': False,
                        'range_values': ['food', 'travel'],
                    },
                    'rating': {
                        'sdtype': 'ordinal',
                        'range_is_nullable': False,
                        'range_values': [1, 2, 3, 4, 5],
                    },
                    'amount': {
                        'sdtype': 'numerical',
                        'range_is_nullable': False,
                        'range_min': 10.5,
                        'range_max': 59.5,
                        'decimal_places': 1,
                    },
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'child_table_name': 'transactions',
                'parent_primary_key': 'user_id',
                'child_foreign_key': 'user_id',
            },
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }

    # Run
    metadata = Metadata.detect_from_dataframes(
        data,
        foreign_key_inference_algorithm='column_name_match',
    )

    # Assert
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_verbose_no_pk_found(capsys):
    """Test 'detect_from_dataframes' verbose output when no PK found."""
    # Setup
    data = {
        'users': pd.DataFrame({'date': pd.date_range(start='2023-01-01', end='2023-01-10')}),
    }
    expected_output = (
        "\nDetecting table 'users':\n"
        "- Column 'date': sdtype='datetime', range_is_nullable=False, "
        "range_min='2023-01-01 00:00:00', range_max='2023-01-10 00:00:00'\n"
        "\nDetecting primary key for table 'users':\n"
        '- No primary key found\n'
        '\nDetecting foreign keys:\n'
        '- No foreign keys found\n'
    )

    # Run
    metadata = Metadata.detect_from_dataframes(
        data, foreign_key_inference_algorithm='column_name_match', verbose=True
    )

    # Assert
    captured = capsys.readouterr().out
    assert expected_output == captured
    assert metadata.tables['users'].primary_key is None


def test_multi_table_metadata():
    """Test ``to_dict`` method on ``Metadata``."""
    # Create an instance
    instance = Metadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {'tables': {}, 'relationships': [], 'METADATA_SPEC_VERSION': 'V2'}
    assert instance.tables == {}
    assert instance.relationships == []


def test_multi_table_metadata_composite_keys():
    """Test ``Metadata`` with composite keys."""
    # Setup
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {
                    'table1_id': {'sdtype': 'id'},
                    'cat_col': {'sdtype': 'categorical'},
                },
                'primary_key': ['table1_id', 'cat_col'],
            },
            'table2': {
                'columns': {
                    'pk': {'sdtype': 'id'},
                    'fk1': {'sdtype': 'id'},
                    'fk2': {'sdtype': 'categorical'},
                },
                'primary_key': 'pk',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': ['table1_id', 'cat_col'],
                'child_table_name': 'table2',
                'child_foreign_key': ['fk1', 'fk2'],
            },
        ],
    }

    # Run
    instance = Metadata.load_from_dict(metadata_dict)
    result = instance.to_dict()

    # Assert
    instance.validate()
    assert result == {**metadata_dict, 'METADATA_SPEC_VERSION': 'V2'}
    assert instance.relationships == metadata_dict['relationships']


@patch('rdt.transformers')
def test_add_column_relationship(mock_rdt_transformers):
    """Test ``add_column_relationship`` method."""

    # Setup
    class RandomLocationGeneratorMock:
        @classmethod
        def _validate_sdtypes(cls, columns_to_sdtypes):
            pass

    mock_rdt_transformers.address.RandomLocationGenerator = RandomLocationGeneratorMock
    _, instance = download_demo('multi_table', 'fake_hotels')
    instance.update_column('city', 'hotels', sdtype='city')
    instance.update_column('state', 'hotels', sdtype='state')

    # Run
    instance.add_column_relationship('address', ['city', 'state'], 'hotels')

    # Assert
    instance.validate()
    assert instance.tables['hotels'].column_relationships == [
        {'type': 'address', 'column_names': ['city', 'state']}
    ]


def test_remove_primary_key():
    # Setup
    metadata = get_multi_table_metadata()

    # Run
    metadata.remove_primary_key('nesreca')

    # Assert
    expected_relationships = [
        {
            'parent_table_name': 'upravna_enota',
            'parent_primary_key': 'id_upravna_enota',
            'child_table_name': 'nesreca',
            'child_foreign_key': 'upravna_enota',
        },
        {
            'parent_table_name': 'upravna_enota',
            'parent_primary_key': 'id_upravna_enota',
            'child_table_name': 'oseba',
            'child_foreign_key': 'upravna_enota',
        },
    ]
    assert metadata.tables['nesreca'].primary_key is None
    assert metadata.relationships == expected_relationships


def test_upgrade_metadata(tmp_path):
    """Test the ``upgrade_metadata`` method."""
    # Setup
    old_metadata = {
        'tables': {
            'nesreca': {
                'fields': {
                    'upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'upravna_enota', 'field': 'id_upravna_enota'},
                    },
                    'id_nesreca': {'type': 'id', 'subtype': 'integer'},
                },
                'primary_key': 'id_nesreca',
            },
            'oseba': {
                'fields': {
                    'upravna_enota': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'upravna_enota', 'field': 'id_upravna_enota'},
                    },
                    'id_nesreca': {
                        'type': 'id',
                        'subtype': 'integer',
                        'ref': {'table': 'nesreca', 'field': 'id_nesreca'},
                    },
                },
            },
            'upravna_enota': {
                'fields': {'id_upravna_enota': {'type': 'id', 'subtype': 'integer'}},
                'primary_key': 'id_upravna_enota',
            },
        }
    }
    filepath = tmp_path / 'old.json'
    old_metadata_file = open(filepath, 'w')
    json.dump(old_metadata, old_metadata_file)
    old_metadata_file.close()

    # Run
    new_metadata = Metadata.upgrade_metadata(filepath=filepath).to_dict()

    # Assert
    expected_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                },
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                    'id_nesreca': {'sdtype': 'id', 'regex_format': r'\d{30}'},
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {'id_upravna_enota': {'sdtype': 'id', 'regex_format': r'\d{30}'}},
            },
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca',
            },
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    assert new_metadata['METADATA_SPEC_VERSION'] == expected_metadata['METADATA_SPEC_VERSION']
    assert new_metadata['tables'] == expected_metadata['tables']
    for relationship in new_metadata['relationships']:
        assert relationship in expected_metadata['relationships']


def test_detect_from_dataframes():
    """Test the ``detect_from_dataframes`` method."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data)

    # Assert
    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'city', 'pii': True},
                    'state': {'sdtype': 'administrative_unit', 'pii': True},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            },
            'guests': {
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id'},
                    'has_rewards': {'sdtype': 'categorical'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical'},
                    'billing_address': {'sdtype': 'categorical'},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
                'primary_key': 'guest_email',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'parent_primary_key': 'hotel_id',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'V2',
    }
    compare_ranges(metadata, real_data)
    compare_metadata(metadata, expected_metadata)


def test_get_column_names():
    """Test the ``get_column_names`` method."""
    # Setup
    metadata = get_multi_table_metadata()

    # Run
    matches = metadata.get_column_names('nesreca', sdtype='id')

    # Assert
    assert set(matches) == {'upravna_enota', 'id_nesreca'}


def test_get_table_metadata():
    """Test the ``get_table_metadata`` method."""
    # Setup
    metadata = get_multi_table_metadata()
    metadata.add_column('latitude', 'nesreca', sdtype='latitude')
    metadata.add_column('longitude', 'nesreca', sdtype='longitude')
    metadata.add_column_relationship('gps', ['latitude', 'longitude'], 'nesreca')

    # Run
    table_metadata = metadata.get_table_metadata('nesreca')

    # Assert
    assert isinstance(table_metadata, Metadata)
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V2',
        'relationships': [],
        'tables': {
            'nesreca': {
                'column_relationships': [
                    {'column_names': ['latitude', 'longitude'], 'type': 'gps'}
                ],
                'columns': {
                    'id_nesreca': {'sdtype': 'id'},
                    'latitude': {'pii': True, 'sdtype': 'latitude'},
                    'longitude': {'pii': True, 'sdtype': 'longitude'},
                    'nesreca_val': {'sdtype': 'numerical'},
                    'upravna_enota': {'sdtype': 'id'},
                },
                'primary_key': 'id_nesreca',
            }
        },
    }
    assert table_metadata.to_dict() == expected_metadata


def test_add_relationship_matching_composite_primary_key():
    """Test that add_relationship succeeds when parent_primary_key matches the actual PK."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'accounts': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'account_type': {'sdtype': 'id'},
                    'col1': {'sdtype': 'numerical'},
                },
                'primary_key': ['user_id', 'account_type'],
            },
            'transactions': {
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'account_type': {'sdtype': 'id'},
                },
                'primary_key': 'transaction_id',
            },
        },
    })

    # Run
    metadata.add_relationship(
        parent_table_name='accounts',
        child_table_name='transactions',
        parent_primary_key=['account_type', 'user_id'],
        child_foreign_key=['account_type', 'user_id'],
    )

    # Assert
    assert len(metadata.relationships) == 1


@pytest.mark.parametrize(
    'parent_primary_key, child_foreign_key',
    [
        (['user_id', 'account_type'], ['user_id', 'account_type']),
        (
            ['user_id', 'account_type', 'region', 'bogus'],
            ['user_id', 'account_type', 'region', 'bogus'],
        ),
        (['user_id', 'bogus'], ['user_id', 'bogus']),
    ],
    ids=['subset of actual pk', 'superset of actual pk', 'partial overlap'],
)
def test_add_relationship_mismatched_primary_key(parent_primary_key, child_foreign_key):
    """Test that add_relationship raises when parent_primary_key doesn't match the actual PK."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'accounts': {
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'account_type': {'sdtype': 'id'},
                    'region': {'sdtype': 'id'},
                    'bogus': {'sdtype': 'id'},
                    'col1': {'sdtype': 'numerical'},
                },
                'primary_key': ['user_id', 'account_type', 'region'],
            },
            'transactions': {
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'account_type': {'sdtype': 'id'},
                    'region': {'sdtype': 'id'},
                    'bogus': {'sdtype': 'id'},
                },
                'primary_key': 'transaction_id',
            },
        },
    })

    # Run and Assert
    error_msg = re.escape(
        f'Relationship between tables (accounts, transactions) '
        f'has a mismatched primary key {sorted(parent_primary_key)}.'
    )
    with pytest.raises(InvalidMetadataError, match=error_msg):
        metadata.add_relationship(
            parent_table_name='accounts',
            child_table_name='transactions',
            parent_primary_key=parent_primary_key,
            child_foreign_key=child_foreign_key,
        )
