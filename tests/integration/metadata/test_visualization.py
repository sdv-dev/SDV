import pandas as pd

from sdv.metadata import MultiTableMetadata, SingleTableMetadata


def test_visualize_graph_for_single_table():
    """Test it runs when a column name contains `>`."""
    # Setup
    data = pd.DataFrame({'>': ['a', 'b', 'c']})
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Run
    metadata.visualize()


def test_visualize_graph_for_multi_table():
    """Test it runs when a column name contains `>`."""
    # Setup
    data1 = pd.DataFrame({'>': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'>': ['a', 'b', 'c']})
    tables = {'1': data1, '2': data2}
    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(tables)
    metadata.update_column('1', '>', sdtype='id')
    metadata.update_column('2', '>', sdtype='id')
    metadata.set_primary_key('1', '>')
    metadata.add_relationship('1', '2', '>', '>')

    # Run
    metadata.visualize()
