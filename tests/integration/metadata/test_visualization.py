import pandas as pd

from sdv.metadata.metadata import Metadata
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer


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
    metadata.update_column('1', '\\|=/bla@#$324%^,"&*()><...', sdtype='id')
    metadata.update_column('2', '\\|=/bla@#$324%^,"&*()><...', sdtype='id')
    metadata.set_primary_key('1', '\\|=/bla@#$324%^,"&*()><...')
    metadata.add_relationship(
        '1', '2', '\\|=/bla@#$324%^,"&*()><...', '\\|=/bla@#$324%^,"&*()><...'
    )
    model = HMASynthesizer(metadata)

    # Run
    metadata.visualize()
    metadata.validate()
    model.fit(tables)
    model.sample(10)
