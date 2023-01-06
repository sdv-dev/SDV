import numpy as np
import pytest
import pandas as pd
from faker import Faker

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


def test_hma_reset_sampling(tmpdir):
    """End to end integration test that uses ``reset_sampling``.

    This test uses ``reset_sampling`` to ensure that the model will generate the same data
    as the first sample after this method has been called.
    """
    # Setup
    faker = Faker()
    metadata, data = load_demo(metadata=True)
    metadata.to_json(tmpdir / 'old.json')
    MultiTableMetadata.upgrade_metadata(tmpdir / 'old.json', tmpdir / 'new.json')
    metadata = MultiTableMetadata.load_from_json(tmpdir / 'new.json')
    metadata.add_column(
        'transactions',
        'ssn',
        sdtype='ssn',
    )
    data['transactions']['ssn'] = [faker.lexify() for _ in range(len(data['transactions']))]

    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.fit(data)
    first_sample = hmasynthesizer.sample()
    second_sample = hmasynthesizer.sample()
    hmasynthesizer.reset_sampling()
    reset_first_sample = hmasynthesizer.sample()
    reset_second_sample = hmasynthesizer.sample()

    # Assert
    for table, reset_table in zip(first_sample.values(), reset_first_sample.values()):
        pd.testing.assert_frame_equal(table, reset_table)

    for table, reset_table in zip(second_sample.values(), reset_second_sample.values()):
        pd.testing.assert_frame_equal(table, reset_table)

    for sample_1, sample_2 in zip(first_sample.values(), second_sample.values()):
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(sample_1, sample_2)
