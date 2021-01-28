from unittest.mock import MagicMock, Mock, call, patch

import graphviz
import pytest

from sdv.metadata import Metadata, visualization


def test__get_graphviz_extension_path_without_extension():
    """Raises a ValueError when the path doesn't contains an extension."""
    with pytest.raises(ValueError):
        visualization._get_graphviz_extension('/some/path')


def test__get_graphviz_extension_invalid_extension():
    """Raises a ValueError when the path contains an invalid extension."""
    with pytest.raises(ValueError):
        visualization._get_graphviz_extension('/some/path.foo')


def test__get_graphviz_extension_none():
    """Get graphviz with path equals to None."""
    # Run
    result = visualization._get_graphviz_extension(None)

    # Asserts
    assert result == (None, None)


def test__get_graphviz_extension_valid():
    """Get a valid graphviz extension."""
    # Run
    result = visualization._get_graphviz_extension('/some/path.png')

    # Asserts
    assert result == ('/some/path', 'png')


def test__add_nodes():
    """Add nodes into a graphviz digraph."""
    # Setup
    metadata = MagicMock(spec_set=Metadata)
    minimock = Mock()

    # pass tests in python3.5
    minimock.items.return_value = (
        ('a_field', {'type': 'numerical', 'subtype': 'integer'}),
        ('b_field', {'type': 'id'}),
        ('c_field', {'type': 'id', 'ref': {'table': 'other', 'field': 'pk_field'}})
    )

    metadata.get_tables.return_value = ['demo']
    metadata.get_fields.return_value = minimock

    metadata.get_primary_key.return_value = 'b_field'
    metadata.get_parents.return_value = set(['other'])
    metadata.get_foreign_keys.return_value = ['c_field']

    metadata.get_table_meta.return_value = {'path': None}

    plot = Mock()

    # Run
    visualization._add_nodes(metadata, plot, names=True)

    # Asserts
    expected_node_label = r"{demo|a_field : numerical - integer\lb_field : id\l" \
                          r"c_field : id\l|Primary key: b_field\l" \
                          r"Foreign key (other): c_field\l}"

    metadata.get_fields.assert_called_once_with('demo')
    metadata.get_primary_key.assert_called_once_with('demo')
    metadata.get_parents.assert_called_once_with('demo')
    metadata.get_table_meta.assert_called_once_with('demo')
    metadata.get_foreign_keys.assert_called_once_with('other', 'demo')
    metadata.get_table_meta.assert_called_once_with('demo')

    plot.node.assert_called_once_with('demo', label=expected_node_label)


def test__add_edges():
    """Add edges into a graphviz digraph."""
    # Setup
    metadata = MagicMock(spec_set=Metadata)

    metadata.get_tables.return_value = ['demo', 'other']
    metadata.get_parents.side_effect = [set(['other']), set()]

    metadata.get_foreign_keys.return_value = ['fk']
    metadata.get_primary_key.return_value = 'pk'

    plot = Mock()

    # Run
    visualization._add_edges(metadata, plot, names=True)

    # Asserts
    expected_edge_label = '   {}.{} > {}.{}'.format('demo', 'fk', 'other', 'pk')

    metadata.get_tables.assert_called_once_with()
    metadata.get_foreign_keys.assert_called_once_with('other', 'demo')
    metadata.get_primary_key.assert_called_once_with('other')
    assert metadata.get_parents.call_args_list == [call('demo'), call('other')]

    plot.edge.assert_called_once_with(
        'other',
        'demo',
        label=expected_edge_label,
        arrowhead='oinv'
    )


@patch('sdv.metadata.visualization.graphviz.Digraph', spec_set=graphviz.Digraph)
@patch('sdv.metadata.visualization._add_nodes', spec_set=visualization._add_nodes)
@patch('sdv.metadata.visualization._add_edges', spec_set=visualization._add_edges)
def test_visualize(add_nodes_mock, add_edges_mock, digraph_mock):
    """Metadata visualize digraph"""
    # Setup
    # plot = Mock(spec_set=graphviz.Digraph)
    # graphviz_mock.Digraph.return_value = plot

    metadata = MagicMock(spec_set=Metadata)

    # Run
    visualization.visualize(metadata, path='output.png')

    # Asserts
    digraph = digraph_mock.return_value
    add_nodes_mock.assert_called_once_with(metadata, digraph, True)
    add_edges_mock.assert_called_once_with(metadata, digraph, True)

    digraph.render.assert_called_once_with(filename='output', cleanup=True, format='png')
