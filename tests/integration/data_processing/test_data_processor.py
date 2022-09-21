"""Integration tests for the ``DataProcessor``."""

import numpy as np

from sdv import load_demo
from sdv.data_processing import DataProcessor
from sdv.metadata import SingleTableMetadata


def test_data_processor_with_anonymized_columns(tmpdir):
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

    Side effects:
        - The column ``occupation`` has been dropped in the ``transform`` process.
        - The column ``occupation`` has been re-created with new values from the
          ``AnonymizedFaker`` instance.
    """
    # Load metadata and data
    metadata, data = load_demo('adult', metadata=True)
    data = data['adult']
    metadata.to_json(tmpdir / 'adult_old.json')
    SingleTableMetadata.upgrade_metadata(tmpdir / 'adult_old.json', tmpdir / 'adult_new.json')

    adult_metadata = SingleTableMetadata.load_from_json(tmpdir / 'adult_new.json')

    # Add anonymized field
    adult_metadata.update_column('occupation', sdtype='job', pii=True)

    # Instance ``DataProcessor``
    dp = DataProcessor(adult_metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
    assert 'occupation' not in transformed.columns


def test_data_processor_with_anonymized_columns_and_primary_key(tmpdir):
    """Test the ``DataProcessor``.

    Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
    a new set of data for that column, while it drops it on the ``transform`` and this values
    can be repeated. Meanwhile, when we set a primary key this has to have a unique length
    of the input data.

    Setup:
        - Load metadata and data.
        - Parse the old metadata to a new metadata.
        - Anonymize the field ``occupation``.
        - Create ``id`` and ``ssn`` columns.

    Run:
        - Create a ``DataProcessor`` with the new metadata.
        - Fit the ``DataProcessor`` instance.
        - Transform the data.
        - Reverse transform the data.

    Side effects:
        - The column ``occupation`` has been dropped in the ``transform`` process.
        - The column ``occupation`` has been re-created with new values from the
          ``AnonymizedFaker`` instance.
        - The column ``id`` has been created in ``transform`` with a unique length of the data.
        - The column ``ssn`` has been created in ``transform`` with a unique length of the data.
    """
    # Load metadata and data
    metadata, data = load_demo('adult', metadata=True)
    data = data['adult']
    metadata.to_json(tmpdir / 'adult_old.json')
    SingleTableMetadata.upgrade_metadata(tmpdir / 'adult_old.json', tmpdir / 'adult_new.json')

    adult_metadata = SingleTableMetadata.load_from_json(tmpdir / 'adult_new.json')

    # Add anonymized field
    adult_metadata.update_column('occupation', sdtype='job', pii=True)

    # Add primary key field
    adult_metadata.add_column('id', sdtype='text', regex_format='ID_\\d{4}[0-9]')
    adult_metadata.set_primary_key('id')

    # Add id
    size = len(data)
    data['id'] = np.arange(0, size).astype('O')

    # Instance ``DataProcessor``
    dp = DataProcessor(adult_metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    reverse_transformed = dp.reverse_transform(transformed)

    # Assert
    assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
    assert 'occupation' not in transformed.columns
    assert transformed.id.isin(data.id).sum() == 0
    assert len(transformed.id.unique()) == size


def test_data_processor_with_primary_key_numerical(tmpdir):
    """Test the ``DataProcessor``.

    Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
    a new set of data for that column, while it drops it on the ``transform`` and this values
    can be repeated. Meanwhile, when we set a primary key this has to have a unique length
    of the input data.

    Setup:
        - Load metadata and data.
        - Parse the old metadata to a new metadata.
        - Anonymize the field ``occupation``.
        - Create ``id`` and ``ssn`` columns.

    Run:
        - Create a ``DataProcessor`` with the new metadata.
        - Fit the ``DataProcessor`` instance.
        - Transform the data.
        - Reverse transform the data.

    Side effects:
        - The column ``occupation`` has been dropped in the ``transform`` process.
        - The column ``occupation`` has been re-created with new values from the
          ``AnonymizedFaker`` instance.
        - The column ``id`` has been created in ``transform`` with a unique length of the data.
        - The column ``ssn`` has been created in ``transform`` with a unique length of the data.
    """
    # Load metadata and data
    data = load_demo('adult')['adult']
    adult_metadata = SingleTableMetadata()
    adult_metadata.detect_from_dataframe(data=data)

    # Add primary key field
    adult_metadata.add_column('id', sdtype='numerical')
    adult_metadata.set_primary_key('id')

    # Add id
    size = len(data)
    data['id'] = np.arange(0, size)

    # Instance ``DataProcessor``
    dp = DataProcessor(adult_metadata)

    # Fit
    dp.fit(data)

    # Transform
    transformed = dp.transform(data)

    # Reverse Transform
    dp.reverse_transform(transformed)

    # Assert
    assert transformed.id.isin(data.id).sum() == size
    assert len(transformed.id.unique()) == size
