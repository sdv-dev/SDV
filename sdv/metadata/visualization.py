"""Functions for Metadata visualization."""

import warnings

import graphviz


def _get_graphviz_extension(path):
    if path:
        path_splitted = path.split('.')
        if len(path_splitted) == 1:
            raise ValueError('Path without graphviz extension.')

        graphviz_extension = path_splitted[-1]

        if graphviz_extension not in graphviz.FORMATS:
            raise ValueError(f'"{graphviz_extension}" not a valid graphviz extension format.')

        return '.'.join(path_splitted[:-1]), graphviz_extension

    return None, None


def visualize_graph(nodes, edges, path=None):
    """Plot metadata usign graphviz.

    Try to generate a plot using graphviz.
    If a ``path`` is provided save the output into a file.

    Args:
        nodes (dict):
            Dictionary mapping a node name to a node label.
        edges (list):
            List of tuples of the format (parent, child, label).
        path (str):
            Output file path to save the plot, it requires a graphviz
            supported extension. If ``None`` do not save the plot.
            Defaults to ``None``.
    """
    try:
        filename, graphviz_extension = _get_graphviz_extension(path)
    except ValueError:
        raise ValueError(
            'Unable to save a visualization with this file type. Try a supported file type like '
            "'png', 'jpg' or 'pdf'. For a full list, see 'https://graphviz.org/docs/outputs/'"
        )

    digraph = graphviz.Digraph(
        'Metadata',
        format=graphviz_extension,
        node_attr={
            'shape': 'Mrecord',
            'fillcolor': 'lightgoldenrod1',
            'style': 'filled'
        },
    )

    for name, label in nodes.items():
        digraph.node(name, label=label)

    for parent, child, label in edges:
        digraph.edge(parent, child, label=label, arrowhead='oinv')

    if filename:
        digraph.render(filename=filename, cleanup=True, format=graphviz_extension)
    else:
        try:
            graphviz.version()
        except graphviz.ExecutableNotFound:
            warning_message = (
                'Graphviz does not seem to be installed on this system. For full '
                'metadata visualization capabilities, please make sure to have its '
                'binaries propertly installed: https://graphviz.gitlab.io/download/'
            )
            warnings.warn(warning_message, RuntimeWarning)

    return digraph
