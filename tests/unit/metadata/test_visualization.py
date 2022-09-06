from unittest.mock import MagicMock, Mock, call, patch

import graphviz
import pytest

from sdv.metadata import Metadata, visualization


def test__get_graphviz_extension_path_without_extension():
    """Raises a ValueError when the path doesn't contains an extension."""
    err_msg = 'Path without graphviz extension.'
    with pytest.raises(ValueError, match=err_msg):
        visualization._get_graphviz_extension('/some/path')


def test__get_graphviz_extension_invalid_extension():
    """Raises a ValueError when the path contains an invalid extension."""
    err_msg = '"foo" not a valid graphviz extension format.'
    with pytest.raises(ValueError, match=err_msg):
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
    metadata.get_parents.return_value = {'other'}
    metadata.get_foreign_keys.return_value = ['c_field']

    metadata.get_table_meta.return_value = {'path': None}

    plot = Mock()

    # Run
    visualization._add_nodes(metadata, plot, names=True, details=True)

    # Asserts
    expected_node_label = (
        r'{demo|a_field : numerical - integer\lb_field : id\l'
        r'c_field : id\l|Primary key: b_field\l'
        r'Foreign key (other): c_field\l}'
    )

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
    metadata.get_parents.side_effect = [{'other'}, {}]

    metadata.get_foreign_keys.return_value = ['fk']
    metadata.get_primary_key.return_value = 'pk'

    plot = Mock()

    # Run
    visualization._add_edges(metadata, plot, names=True, details=True)

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
    metadata = MagicMock(spec_set=Metadata)

    # Run
    visualization.visualize(metadata, path='output.png')

    # Asserts
    digraph = digraph_mock.return_value
    add_nodes_mock.assert_called_once_with(metadata, digraph, True, True)
    add_edges_mock.assert_called_once_with(metadata, digraph, True, True)

    digraph.render.assert_called_once_with(filename='output', cleanup=True, format='png')


@patch('sdv.metadata.visualization.graphviz.Digraph', spec_set=graphviz.Digraph)
def test_visualize_graph(digraph_mock):
    """Test the ``visualize_graph`` method.

    Setup:
        - Mock the ``graphviz.Digraph`` object.

    Input:
        - nodes set to a dictionary with one node.
        - edges set to a list with one edge.
        - path set to a filepath

    Side effect:
        - The digraph_mock should add the edge and node and call ``render`` with
        the correct parameters.
    """
    # Setup
    digraph = digraph_mock.return_value
    nodes = {'node': 'node label'}
    edges = [('node1', 'node2', 'edge label')]

    # Run
    result = visualization.visualize_graph(nodes=nodes, edges=edges, path='output.png')

    # Asserts
    digraph.edge.assert_called_once_with('node1', 'node2', label='edge label', arrowhead='oinv')
    digraph.node.assert_called_once_with('node', label='node label')
    digraph.render.assert_called_once_with(filename='output', cleanup=True, format='png')
    assert result == digraph


@patch('sdv.metadata.visualization.graphviz.Digraph', spec_set=graphviz.Digraph)
def test_visualize_graph_no_path(digraph_mock):
    """Test the ``visualize_graph`` method.

    If no path is provided, the digraph should not be rendered.

    Setup:
        - Mock the ``graphviz.Digraph`` object.

    Input:
        - nodes set to a dictionary with one node.
        - edges set to a list with one edge.
        - path set to None

    Side effect:
        - The digraph_mock should add the edge and node.
    """
    # Setup
    digraph = digraph_mock.return_value
    nodes = {'node': 'node label'}
    edges = [('node1', 'node2', 'edge label')]

    # Run
    result = visualization.visualize_graph(nodes=nodes, edges=edges)

    # Asserts
    digraph.edge.assert_called_once_with('node1', 'node2', label='edge label', arrowhead='oinv')
    digraph.node.assert_called_once_with('node', label='node label')
    digraph.render.assert_not_called()
    assert result == digraph


@patch('sdv.metadata.visualization.graphviz.version')
@patch('sdv.metadata.visualization.graphviz.Digraph', spec_set=graphviz.Digraph)
@patch('sdv.metadata.visualization.warnings')
def test_visualize_graph_no_path_graphviz_not_installed(warnings_mock, digraph_mock, version_mock):
    """Test the ``visualize_graph`` method.

    If graphviz is not installed, raise a warning.

    Setup:
        - Mock the ``graphviz.Digraph`` object.
        - Mock warnings.
        - Set the side effect for ``version`` to raise a ``graphviz.ExecutableNotFound`` error.

    Input:
        - nodes set to a dictionary with one node.
        - edges set to a list with one edge.
        - path set to None

    Side effect:
        - A warning should be raised.
    """
    # Setup
    digraph = digraph_mock.return_value
    nodes = {'node': 'node label'}
    edges = [('node1', 'node2', 'edge label')]
    version_mock.side_effect = Mock(side_effect=graphviz.ExecutableNotFound(['version']))

    # Run
    result = visualization.visualize_graph(nodes=nodes, edges=edges)

    # Asserts
    warning = (
        'Graphviz does not seem to be installed on this system. For full '
        'metadata visualization capabilities, please make sure to have its '
        'binaries propertly installed: https://graphviz.gitlab.io/download/'
    )
    digraph.edge.assert_called_once_with('node1', 'node2', label='edge label', arrowhead='oinv')
    digraph.node.assert_called_once_with('node', label='node label')
    digraph.render.assert_not_called()
    assert result == digraph
    warnings_mock.warn.assert_called_once_with(warning, RuntimeWarning)


@patch('sdv.metadata.visualization._get_graphviz_extension')
def test_visualize_graph_bad_extension(extension_mock):
    """Test the ``visualize_graph`` method.

    If the file extension is bad, an error should be raised.

    Setup:
        - Mock the ``_get_graphviz_extension`` to raise an error.

    Input:
        - nodes set to a dictionary with one node.
        - edges set to a list with one edge.
        - path set to a string with no extension.

    Side effect:
        - A ``ValueError`` should be raised.
    """
    # Setup
    extension_mock.side_effect = ValueError()
    nodes = {'node': 'node label'}
    edges = [('node1', 'node2', 'edge label')]

    # Run
    error_message = (
        'Unable to save a visualization with this file type. Try a supported file type like '
        "'png', 'jpg' or 'pdf'. For a full list, see 'https://graphviz.org/docs/outputs/'"
    )
    with pytest.raises(ValueError, match=error_message):
        visualization.visualize_graph(nodes=nodes, edges=edges, path='path')
