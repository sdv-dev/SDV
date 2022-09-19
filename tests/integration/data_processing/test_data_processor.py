"""Integration tests for the ``DataProcessor``."""

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
