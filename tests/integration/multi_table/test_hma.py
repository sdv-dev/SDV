from sdv import load_demo
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer


def test_hma(tmpdir):
    """End to end integration tests with ``HMASynthesizer``.

    The test consist on loading the demo data, convert the old metadata to the new format
    and then fit a ``HMASynthesizer``. After fitting two samples are being generated, one with
    a 1.0 scale and one with 2.5 scale.
    """
    # Setup
    metadata, data = load_demo(metadata=True)
    metadata.to_json(tmpdir / 'old.json')
    MultiTableMetadata.upgrade_metadata(tmpdir / 'old.json', tmpdir / 'new.json')
    metadata = MultiTableMetadata.load_from_json(tmpdir / 'new.json')

    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.fit(data)
    normal_sample = hmasynthesizer.sample()
    increased_sample = hmasynthesizer.sample(2.5)

    # Assert
    assert list(normal_sample) == ['users', 'sessions', 'transactions']
    assert list(increased_sample) == ['users', 'sessions', 'transactions']
    for table_name, table in normal_sample.items():
        assert all(table.columns == data[table_name].columns)

    for normal_table, increased_table in zip(normal_sample.values(), increased_sample.values()):
        assert increased_table.size > normal_table.size
