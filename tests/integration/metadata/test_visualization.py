import pandas as pd

from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def test_visualize_graph_for_sequential_data():
    """Test visualization has sequence key and index with sequential data."""
    # Setup
    _, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')

    # Run
    graph = metadata.visualize()

    # Assert
    assert 'Sequence index' in graph.source
    assert 'Sequence key' in graph.source
    assert 'Primary key' in graph.source
    assert 'nasdaq100_2019' in graph.source
    assert 'relationships' not in graph.source


def test_visualize_graph_for_single_table():
    """Test it runs when a column name contains symbols."""
    # Setup
    data = pd.DataFrame({'\\|=/bla@#$324%^,"&*()><...': ['a', 'b', 'c']})
    metadata = Metadata.detect_from_dataframes({'table': data})
    model = GaussianCopulaSynthesizer(metadata)

    # Run
    metadata.visualize()
    metadata.validate()
    model.fit(data)
    model.sample(10)


def test_visualize_graph_for_multi_table():
    """Test it runs when a column name contains symbols."""
    # Setup
    data1 = pd.DataFrame({'\\|=/bla@#$324%^,"&*()><...': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'\\|=/bla@#$324%^,"&*()><...': ['a', 'b', 'c']})
    tables = {'1': data1, '2': data2}
    metadata = Metadata.detect_from_dataframes(tables)
    metadata.update_column('\\|=/bla@#$324%^,"&*()><...', '1', sdtype='id')
    metadata.update_column('\\|=/bla@#$324%^,"&*()><...', '2', sdtype='id')
    metadata.set_primary_key('\\|=/bla@#$324%^,"&*()><...', '1')
    metadata.add_relationship(
        '1', '2', '\\|=/bla@#$324%^,"&*()><...', '\\|=/bla@#$324%^,"&*()><...'
    )
    model = HMASynthesizer(metadata)

    # Run
    metadata.visualize()
    metadata.validate()
    model.fit(tables)
    model.sample(10)


def test_visualize_pk_to_pk(primary_key_to_primary_key):
    """Test visualization runs with primary to primary key relationship."""
    # Setup
    _, metadata = primary_key_to_primary_key

    # Run and Assert
    metadata.visualize()
