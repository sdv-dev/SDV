import os
import re

import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer

DEFAULT_TABLE_NAME = 'default_table_name'


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


def test_detect_from_dataframes_multi_table():
    """Test the ``detect_from_dataframes`` method works with multi-table."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata()

    # Run
    metadata.detect_from_dataframes(real_data)

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
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
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


def test_detect_from_data_frames_single_table():
    """Test the ``detect_from_dataframes`` method works with a single table."""
    # Setup
    data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata()
    metadata.detect_from_dataframes({'table_1': data['hotels']})

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
                    'classification': {'sdtype': 'unknown', 'pii': True},
                },
                'primary_key': 'hotel_id',
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
                    'billing_address': {'sdtype': 'unknown', 'pii': True},
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


def test_detect_table_from_csv(tmp_path):
    """Test the ``detect_table_from_csv`` method."""
    # Setup
    real_data, _ = download_demo(modality='multi_table', dataset_name='fake_hotels')

    metadata = Metadata()

    for table_name, dataframe in real_data.items():
        csv_path = tmp_path / f'{table_name}.csv'
        dataframe.to_csv(csv_path, index=False)

    # Run
    metadata.detect_table_from_csv('hotels', tmp_path / 'hotels.csv')

    # Assert
    metadata.update_column(
        table_name='hotels',
        column_name='city',
        sdtype='categorical',
    )
    metadata.update_column(
        table_name='hotels',
        column_name='state',
        sdtype='categorical',
    )
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
                    'city': {'sdtype': 'categorical'},
                    'state': {'sdtype': 'categorical'},
                    'rating': {'sdtype': 'numerical'},
                    'classification': {'sdtype': 'categorical'},
                },
                'primary_key': 'hotel_id',
            }
        },
        'relationships': [],
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
