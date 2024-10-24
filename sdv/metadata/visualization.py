"""Functions for Metadata visualization."""

import warnings
from collections import defaultdict

import graphviz

DEFAULT_SDTYPES = ['id', 'numerical', 'categorical', 'datetime', 'boolean']


def create_columns_node(columns):
    """Convert columns into text for ``graphviz`` node.

    Args:
        columns (dict):
            A dict mapping the column names with a dictionary containing the ``sdtype`` of the
            column name.

    Returns:
        str:
            String representing the node that will be printed for the given columns.
    """
    columns = [rf'{name} : {meta.get("sdtype")}' for name, meta in columns.items()]
    return r'\l'.join(columns)


def create_summarized_columns_node(columns):
    """Convert columns into summarized text for ``graphviz`` node.

    Args:
        columns (dict):
            A dict mapping the column names with a dictionary containing the ``sdtype`` of the
            column name.

    Returns:
        str:
            String representing the node that will be printed for the given columns.
    """
    count_dict = defaultdict(int)
    for column_name, meta in columns.items():
        sdtype = 'other' if meta['sdtype'] not in DEFAULT_SDTYPES else meta['sdtype']
        count_dict[sdtype] += 1

    count_dict = dict(sorted(count_dict.items()))
    columns = ['Columns']
    columns.extend([rf'&nbsp; &nbsp; • {sdtype} : {count}' for sdtype, count in count_dict.items()])

    return r'\l'.join(columns)


def _get_graphviz_extension(filepath):
    if filepath:
        path_splitted = filepath.split('.')
        if len(path_splitted) == 1:
            raise ValueError('Path without graphviz extension.')

        graphviz_extension = path_splitted[-1]

        if graphviz_extension not in graphviz.FORMATS:
            raise ValueError(f'"{graphviz_extension}" not a valid graphviz extension format.')

        return '.'.join(path_splitted[:-1]), graphviz_extension

    return None, None


def _replace_special_characters(string):
    return string.replace('<', '\\<').replace('>', '\\>')  # noqa: W605


def visualize_graph(nodes, edges, filepath=None):
    """Plot metadata using graphviz.

    Try to generate a plot using graphviz.
    If a ``filepath`` is provided save the output into a file.

    Args:
        nodes (dict):
            Dictionary mapping a node name to a node label.
        edges (list):
            List of tuples of the format (parent, child, label).
        filepath (str):
            Output file path to save the plot, it requires a graphviz
            supported extension. If ``None`` do not save the plot.
            Defaults to ``None``.
    """
    try:
        filename, graphviz_extension = _get_graphviz_extension(filepath)
    except ValueError:
        raise ValueError(
            'Unable to save a visualization with this file type. Try a supported file type like '
            "'png', 'jpg' or 'pdf'. For a full list, see 'https://graphviz.org/docs/outputs/'"
        )

    digraph = graphviz.Digraph(
        'Metadata',
        format=graphviz_extension,
        node_attr={'shape': 'Mrecord', 'fillcolor': 'lightgoldenrod1', 'style': 'filled'},
    )

    for name, label in nodes.items():
        digraph.node(name, label=_replace_special_characters(label))

    for parent, child, label in edges:
        digraph.edge(parent, child, label=_replace_special_characters(label), arrowhead='oinv')

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
