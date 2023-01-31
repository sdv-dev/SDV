import datetime

import pandas as pd
import pkg_resources
import pytest
from faker import Faker

from sdv.datasets.demo import download_demo
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table import HMASynthesizer


def test_hma(tmpdir):
    """End to end integration tests with ``HMASynthesizer``.

    The test consist on loading the demo data, convert the old metadata to the new format
    and then fit a ``HMASynthesizer``. After fitting two samples are being generated, one with
    a 0.5 scale and one with 1.5 scale.
    """
    # Setup
    data, metadata = download_demo('multi_table', 'got_families')
    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.fit(data)
    normal_sample = hmasynthesizer.sample(0.5)
    increased_sample = hmasynthesizer.sample(1.5)

    # Assert
    assert list(normal_sample) == ['characters', 'character_families', 'families']
    assert list(increased_sample) == ['characters', 'character_families', 'families']
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
    data, metadata = download_demo('multi_table', 'got_families')
    metadata.add_column(
        'characters',
        'ssn',
        sdtype='ssn',
    )
    data['characters']['ssn'] = [faker.lexify() for _ in range(len(data['characters']))]
    for table in metadata._tables.values():
        table._alternate_keys = []

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


def test_get_info():
    """Test the correct dictionary is returned.

    Check the return dictionary is valid both before and after fitting the synthesizer.
    """
    # Setup
    data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    metadata = MultiTableMetadata()
    metadata.add_table('tab')
    metadata.add_column('tab', 'col', sdtype='numerical')
    synthesizer = HMASynthesizer(metadata)

    # Run
    info = synthesizer.get_info()

    # Assert
    assert info == {
        'class_name': 'HMASynthesizer',
        'creation_date': today,
        'is_fit': False,
        'last_fit_date': None,
        'fitted_sdv_version': None
    }

    # Run
    synthesizer.fit(data)
    info = synthesizer.get_info()

    # Assert
    version = pkg_resources.get_distribution('sdv').version
    assert info == {
        'class_name': 'HMASynthesizer',
        'creation_date': today,
        'is_fit': True,
        'last_fit_date': today,
        'fitted_sdv_version': version
    }


def test_hma_set_parameters():
    """Test the ``set_table_parameters``.

    Validate that the ``set_table_parameters`` sets new parameters to the synthesizers.
    """
    # Setup
    data, metadata = download_demo('multi_table', 'got_families')
    hmasynthesizer = HMASynthesizer(metadata)

    # Run
    hmasynthesizer.set_table_parameters('characters', {'default_distribution': 'gamma'})
    hmasynthesizer.set_table_parameters('families', {'default_distribution': 'uniform'})
    hmasynthesizer.set_table_parameters('character_families', {'default_distribution': 'norm'})

    # Assert
    assert hmasynthesizer.get_table_parameters('characters') == {'default_distribution': 'gamma'}
    assert hmasynthesizer.get_table_parameters('families') == {'default_distribution': 'uniform'}
    assert hmasynthesizer.get_table_parameters('character_families') == {
        'default_distribution': 'norm'
    }

    assert hmasynthesizer._table_synthesizers['characters'].default_distribution == 'gamma'
    assert hmasynthesizer._table_synthesizers['families'].default_distribution == 'uniform'
    assert hmasynthesizer._table_synthesizers['character_families'].default_distribution == 'norm'
