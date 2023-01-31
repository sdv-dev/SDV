"""Integration tests for the ``DataProcessor``."""
import itertools

import numpy as np
from rdt.transformers import AnonymizedFaker, FloatFormatter, LabelEncoder, UnixTimestampEncoder

from sdv.data_processing import DataProcessor
from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata


def test_data_processor_with_anonymized_columns():
    """Test the ``DataProcessor``.

    Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
    a new set of data for that column, while it drops it on the ``transform``.

    Setup:
        - Load metadata and data.
        - Parse the old metadata to a new metadata.
        - Anonymize the field ``occupation``.

    Run:
        - Create a ``DataProcessor`` with the new metadata.
        - Fit the ``DataProcessor`` instance.
        - Transform the data.
        - Reverse transform the data.

    Assert:
        - The column ``occupation`` has been dropped in the ``transform`` process.
        - The column ``occupation`` has been re-created with new values from the
          ``AnonymizedFaker`` instance.
    """
    # Load metadata and data
    data, metadata = download_demo('single_table', 'adult')

    # Add anonymized field
    metadata.update_column('occupation', sdtype='job', pii=True)

    # Instance ``DataProcessor``
    dp = DataProcessor(metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
    assert 'occupation' not in transformed.columns


def test_data_processor_with_anonymized_columns_and_primary_key():
    """Test the ``DataProcessor``.

    Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
    a new set of data for that column, while it drops it on the ``transform`` and this values
    can be repeated. Meanwhile, when we set a primary key this has to have a unique length
    of the input data.

    Setup:
        - Load metadata and data.
        - Parse the old metadata to a new metadata.
        - Anonymize the field ``occupation``.
        - Create ``id``.

    Run:
        - Create a ``DataProcessor`` with the new metadata.
        - Fit the ``DataProcessor`` instance.
        - Transform the data.
        - Reverse transform the data.

    Assert:
        - The column ``occupation`` has been dropped in the ``transform`` process.
        - The column ``occupation`` has been re-created with new values from the
          ``AnonymizedFaker`` instance.
        - The column ``id`` has been created in ``transform`` with a unique length of the data.
    """
    # Load metadata and data
    data, metadata = download_demo('single_table', 'adult')

    # Add anonymized field
    metadata.update_column('occupation', sdtype='job', pii=True)

    # Add primary key field
    metadata.add_column('id', sdtype='text', regex_format='ID_\\d{4}[0-9]')
    metadata.set_primary_key('id')

    # Add id
    size = len(data)
    data['id'] = np.arange(0, size).astype('O')

    # Instance ``DataProcessor``
    dp = DataProcessor(metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert transformed.index.name == 'id'
    assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
    assert 'occupation' not in transformed.columns
    assert 'id' not in transformed.columns
    assert reverse_transformed.id.isin(data.id).sum() == 0
    assert len(reverse_transformed.id.unique()) == size


def test_data_processor_with_primary_key_numerical():
    """End to end test for the ``DataProcessor``.

    Test that when running the ``DataProcessor`` with a numerical primary key, this is able
    to generate such a key in the ``transform`` method. The key should be the same length
    as the input data size.

    Setup:
        - Load metadata and data.
        - Detect the metadata from the ``dataframe``.
        - Create ``id`` column.

    Run:
        - Create a ``DataProcessor`` with the new metadata.
        - Fit the ``DataProcessor`` instance.
        - Transform the data.
        - Reverse transform the data.

    Assert:
        - The column ``id`` has been created during ``transform`` with a unique length of the data.
          matching the original numerical ``id`` column.
    """
    # Load metadata and data
    data, _ = download_demo('single_table', 'adult')
    adult_metadata = SingleTableMetadata()
    adult_metadata.detect_from_dataframe(data=data)

    # Add primary key field
    adult_metadata.add_column('id', sdtype='numerical')
    adult_metadata.set_primary_key('id')

    # Add id
    size = len(data)
    id_generator = itertools.count()
    ids = [next(id_generator) for _ in range(size)]
    data['id'] = ids

    # Instance ``DataProcessor``
    dp = DataProcessor(adult_metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert transformed.index.name == 'id'
    assert 'id' not in transformed.columns
    assert reverse_transformed.index.isin(data.id).sum() == size
    assert len(reverse_transformed.id.unique()) == size


def test_data_processor_with_alternate_keys():
    """Test that alternate keys are being generated in a unique way.

    Test that the alternate keys are being generated and dropped the same way
    as with the ``primary_key``.
    """
    # Load metadata and data
    data, _ = download_demo('single_table', 'adult')
    data['fnlwgt'] = data['fnlwgt'].astype(str)
    adult_metadata = SingleTableMetadata()
    adult_metadata.detect_from_dataframe(data=data)

    # Add primary key field
    adult_metadata.add_column('id', sdtype='numerical')
    adult_metadata.set_primary_key('id')

    adult_metadata.add_column('secondary_id', sdtype='numerical')
    adult_metadata.update_column('fnlwgt', sdtype='text', regex_format='ID_\\d{4}[0-9]')

    adult_metadata.add_alternate_keys(['secondary_id', 'fnlwgt'])

    # Add id
    size = len(data)
    id_generator = itertools.count()
    ids = [next(id_generator) for _ in range(size)]
    data['id'] = ids
    data['secondary_id'] = ids

    # Instance ``DataProcessor``
    dp = DataProcessor(adult_metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert 'id' not in transformed.columns
    assert 'secondary_id' not in transformed.columns
    assert 'fnlwgt' not in transformed.columns
    assert len(reverse_transformed.id.unique()) == size
    assert len(reverse_transformed.secondary_id.unique()) == size
    assert len(reverse_transformed.fnlwgt.unique()) == size


def test_data_processor_prepare_for_fitting():
    # Setup
    data, metadata = download_demo(
        modality='single_table',
        dataset_name='student_placements_pii'
    )
    dp = DataProcessor(metadata)

    # Run
    dp.prepare_for_fitting(data)

    # Assert
    field_transformers = dp._hyper_transformer.field_transformers
    expected_transformers = {
        'mba_spec': LabelEncoder,
        'employability_perc': FloatFormatter,
        'placed': LabelEncoder,
        'student_id': None,
        'experience_years': FloatFormatter,
        'duration': LabelEncoder,
        'salary': FloatFormatter,
        'second_perc': FloatFormatter,
        'start_date': UnixTimestampEncoder,
        'address': AnonymizedFaker,
        'gender': LabelEncoder,
        'mba_perc': FloatFormatter,
        'degree_type': LabelEncoder,
        'end_date': UnixTimestampEncoder,
        'high_spec': LabelEncoder,
        'high_perc': FloatFormatter,
        'work_experience': LabelEncoder,
        'degree_perc': FloatFormatter
    }
    for column_name, transformer_class in expected_transformers.items():
        if transformer_class is not None:
            assert isinstance(field_transformers[column_name], transformer_class)
        else:
            assert field_transformers[column_name] is None

    assert field_transformers['start_date'].datetime_format == '%Y-%m-%d'
    assert field_transformers['end_date'].datetime_format == '%Y-%m-%d'
    assert field_transformers['salary'].computer_representation == 'Int64'
