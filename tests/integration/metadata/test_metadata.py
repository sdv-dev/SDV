import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer

DEFAULT_TABLE_NAME = 'table'


def test_metadata():
    """Test ``MultiTableMetadata``."""
    # Create an instance
    instance = Metadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {'tables': {}, 'relationships': [], 'METADATA_SPEC_VERSION': 'V1'}
    assert instance.tables == {}
    assert instance.relationships == []


def test_load_from_json_single_table_metadata(tmp_path):
    """Test the ``load_from_json`` method with a single table metadata."""
    # Setup
    old_metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'column_1': {'sdtype': 'numerical'},
            'column_2': {'sdtype': 'categorical'},
        },
    })
    old_metadata.save_to_json(tmp_path / 'metadata.json')
    expected_warning = re.escape(
        'You are loading an older SingleTableMetadata object. This will be converted '
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
        'METADATA_SPEC_VERSION': 'V1',
    }


def test_detect_from_dataframes_multi_table():
    """Test the ``detect_from_dataframes`` method works with multi-table."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

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
        'METADATA_SPEC_VERSION': 'V1',
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_multi_table_without_infer_sdtypes():
    """Test it when infer_sdtypes is False."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_sdtypes=False)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

    expected_metadata = {
        'tables': {
            'hotels': {
                'columns': {
                    'hotel_id': {'sdtype': 'id'},
                    'city': {'sdtype': 'unknown', 'pii': True},
                    'state': {'sdtype': 'unknown', 'pii': True},
                    'rating': {'sdtype': 'unknown', 'pii': True},
                    'classification': {'sdtype': 'categorical'},
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
        'METADATA_SPEC_VERSION': 'V1',
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_multi_table_with_infer_keys_primary_only():
    """Test it when infer_keys is 'primary_only'."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_keys='primary_only')

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

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
        'METADATA_SPEC_VERSION': 'V1',
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_multi_table_with_infer_keys_none():
    """Test it when infer_keys is None."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    metadata = Metadata.detect_from_dataframes(real_data, infer_keys=None)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

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
        'METADATA_SPEC_VERSION': 'V1',
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_single_table():
    """Test the ``detect_from_dataframes`` method works with a single table."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata.detect_from_dataframes({'table_1': data['hotels']})

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_single_table_infer_sdtypes_false():
    """Test it for a single table when infer_sdtypes is False."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframes({'table_1': data['hotels']}, infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_single_table_infer_keys_primary_only():
    """Test it for a single table when infer_keys is 'primary_only'."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframes(
        {'table_1': data['hotels']}, infer_keys='primary_only'
    )

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframes_single_table_infer_keys_none():
    """Test it for a single table when infer_keys is None."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframes({'table_1': data['hotels']}, infer_keys=None)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframe():
    """Test that a single table can be detected as a DataFrame."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata.detect_from_dataframe(data['hotels'])

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframe_infer_sdtypes_false():
    """Test it when infer_sdtypes is False."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframe(data['hotels'], infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframe_infer_keys_none():
    """Test it when infer_keys is None."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframe(data['hotels'], infer_keys=None)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframe_infer_keys_none_infer_sdtypes_false():
    """Test it when infer_keys is None and infer_sdtypes is False."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')
    metadata = Metadata.detect_from_dataframe(data['hotels'], infer_keys=None, infer_sdtypes=False)

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'V1',
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
    assert metadata.to_dict() == expected_metadata


def test_detect_from_csvs(tmp_path):
    """Test the ``detect_from_csvs`` method."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_from_csvs(folder_name=tmp_path)

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='classification',
        sdtype='categorical',
    )

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
        'METADATA_SPEC_VERSION': 'V1',
    }

    assert metadata.to_dict() == expected_metadata


def test_single_table_compatibility(tmp_path):
    """Test if SingleTableMetadata still has compatibility with single table synthesizers."""
    # Setup
    data, _ = download_demo('single_table', 'fake_hotel_guests')
    warn_msg = (
        "The 'SingleTableMetadata' is deprecated. Please use the new "
        "'Metadata' class for synthesizers."
    )

    single_table_metadata_dict = {
        'primary_key': 'guest_email',
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'guest_email': {'sdtype': 'email', 'pii': True},
            'has_rewards': {'sdtype': 'boolean'},
            'room_type': {'sdtype': 'categorical'},
            'amenities_fee': {'sdtype': 'numerical', 'computer_representation': 'Float'},
            'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
            'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
            'room_rate': {'sdtype': 'numerical', 'computer_representation': 'Float'},
            'billing_address': {'sdtype': 'address', 'pii': True},
            'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
        },
    }
    metadata = SingleTableMetadata.load_from_dict(single_table_metadata_dict)
    assert isinstance(metadata, SingleTableMetadata)

    # Run
    with pytest.warns(FutureWarning, match=warn_msg):
        synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    model_path = os.path.join(tmp_path, 'synthesizer.pkl')
    synthesizer.save(model_path)

    # Assert
    assert os.path.exists(model_path)
    assert os.path.isfile(model_path)
    loaded_synthesizer = GaussianCopulaSynthesizer.load(model_path)
    assert isinstance(synthesizer, GaussianCopulaSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    loaded_sample = loaded_synthesizer.sample(10)
    synthesizer.validate(loaded_sample)

    # Run against Metadata
    synthesizer_2 = GaussianCopulaSynthesizer(Metadata.load_from_dict(metadata.to_dict()))
    synthesizer_2.fit(data)
    metadata_sample = synthesizer.sample(10)
    assert loaded_synthesizer.metadata.to_dict() == synthesizer_2.metadata.to_dict()
    assert metadata_sample.columns.to_list() == loaded_sample.columns.to_list()


def test_multi_table_compatibility(tmp_path):
    """Test if MultiTableMetadata still has compatibility with multi table synthesizers."""
    # Setup
    data, _ = download_demo('multi_table', 'fake_hotels')
    warn_msg = re.escape(
        "The 'MultiTableMetadata' is deprecated. Please use the new "
        "'Metadata' class for synthesizers."
    )

    multi_dict = {
        'tables': {
            'guests': {
                'primary_key': 'guest_email',
                'columns': {
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'hotel_id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'has_rewards': {'sdtype': 'boolean'},
                    'room_type': {'sdtype': 'categorical'},
                    'amenities_fee': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'room_rate': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'billing_address': {'sdtype': 'address', 'pii': True},
                    'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
                },
            },
            'hotels': {
                'primary_key': 'hotel_id',
                'columns': {
                    'hotel_id': {'sdtype': 'id', 'regex_format': 'HID_[0-9]{3}'},
                    'city': {'sdtype': 'categorical'},
                    'state': {'sdtype': 'categorical'},
                    'rating': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'classification': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'hotels',
                'parent_primary_key': 'hotel_id',
                'child_table_name': 'guests',
                'child_foreign_key': 'hotel_id',
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }
    metadata = MultiTableMetadata.load_from_dict(multi_dict)
    assert type(metadata) is MultiTableMetadata

    # Run
    with pytest.warns(FutureWarning, match=warn_msg):
        synthesizer = HMASynthesizer(metadata)

    synthesizer.fit(data)
    model_path = os.path.join(tmp_path, 'synthesizer.pkl')
    synthesizer.save(model_path)

    # Assert
    assert os.path.exists(model_path)
    assert os.path.isfile(model_path)

    # Load HMASynthesizer
    loaded_synthesizer = HMASynthesizer.load(model_path)

    # Asserts
    assert isinstance(synthesizer, HMASynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()

    # Load Metadata
    expected_metadata = metadata.to_dict()

    # Asserts

    assert loaded_synthesizer.metadata.to_dict() == expected_metadata

    # Sample from loaded synthesizer
    loaded_sample = loaded_synthesizer.sample(10)
    synthesizer.validate(loaded_sample)

    # Run against Metadata
    synthesizer_2 = HMASynthesizer(Metadata.load_from_dict(metadata.to_dict()))
    synthesizer_2.fit(data)
    metadata_sample = synthesizer.sample(10)
    expected_metadata = loaded_synthesizer.metadata.to_dict()
    expected_metadata['METADATA_SPEC_VERSION'] = 'V1'
    assert expected_metadata == synthesizer_2.metadata.to_dict()
    for table in metadata_sample:
        assert metadata_sample[table].columns.to_list() == loaded_sample[table].columns.to_list()


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
    _, metadata = download_demo('single_table', 'fake_hotel_guests')
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
    _, metadata = download_demo('multi_table', 'fake_hotels')
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

    assert anonymized.METADATA_SPEC_VERSION == 'V1'
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


def test_no_duplicated_foreign_key_relationships_are_generated():
    # Setup
    parent_a = pd.DataFrame(
        data={
            'id': ['id-' + str(i) for i in range(100)],
            'col1': [round(i, 2) for i in np.random.uniform(low=0, high=10, size=100)],
        }
    )
    parent_b = pd.DataFrame(
        data={
            'id': ['id-' + str(i) for i in range(100)],
            'col2': [round(i, 2) for i in np.random.uniform(low=0, high=10, size=100)],
        }
    )

    child_c = pd.DataFrame(
        data={
            'id': ['id-' + str(i) for i in np.random.randint(0, 100, size=1000)],
            'col3': [round(i, 2) for i in np.random.uniform(low=0, high=10, size=1000)],
        }
    )

    data = {'parent_a': parent_a, 'parent_b': parent_b, 'child_c': child_c}

    # Run
    metadata = Metadata.detect_from_dataframes(data)

    # Assert
    assert metadata.relationships == [
        {
            'parent_table_name': 'parent_a',
            'child_table_name': 'child_c',
            'parent_primary_key': 'id',
            'child_foreign_key': 'id',
        }
    ]


def test_validate_metadata_with_reused_foreign_keys():
    # Setup
    metadata_dict = {
        'tables': {
            'A1': {
                'columns': {
                    'data': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                },
                'primary_key': 'id',
            },
            'A2': {
                'columns': {
                    'data': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'fk1_A1': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                },
                'primary_key': 'id',
            },
            'A3': {
                'columns': {
                    'data': {'sdtype': 'numerical', 'computer_representation': 'Float'},
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
        'Relationship between tables (A2, A3) uses a foreign key column '
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
        'METADATA_SPEC_VERSION': 'V1',
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
        'METADATA_SPEC_VERSION': 'V1',
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
        'METADATA_SPEC_VERSION': 'V1',
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
        'METADATA_SPEC_VERSION': 'V1',
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
        'METADATA_SPEC_VERSION': 'V1',
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
        'METADATA_SPEC_VERSION': 'V1',
    }
    assert expected_metadata_dict == metadata.to_dict()
    assert metadata._multi_table_updated
