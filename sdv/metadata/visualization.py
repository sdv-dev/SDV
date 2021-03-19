"""Functions for Metadata visualization."""

import warnings

import graphviz


def _get_graphviz_extension(path):
    if path:
        path_splitted = path.split('.')
        if len(path_splitted) == 1:
            raise ValueError('Path without graphviz extansion.')

        graphviz_extension = path_splitted[-1]

        if graphviz_extension not in graphviz.backend.FORMATS:
            raise ValueError(
                '"{}" not a valid graphviz extension format.'.format(graphviz_extension)
            )

        return '.'.join(path_splitted[:-1]), graphviz_extension

    return None, None


def _add_nodes(metadata, digraph, names, details):
    """Add nodes into a `graphviz.Digraph`.

    Each node represent a metadata table.

    Args:
        metadata (Metadata):
            Metadata object to plot.
        digraph (graphviz.Digraph):
            graphviz.Digraph being built
    """
    for table in metadata.get_tables():
        if not names:
            title = ''
        elif details:
            # Append table fields
            fields = []

            for name, value in metadata.get_fields(table).items():
                if value.get('subtype') is not None:
                    fields.append('{} : {} - {}'.format(name, value['type'], value['subtype']))
                else:
                    fields.append('{} : {}'.format(name, value['type']))

            fields = r'\l'.join(fields)

            # Append table extra information
            extras = []

            primary_key = metadata.get_primary_key(table)
            if primary_key is not None:
                extras.append('Primary key: {}'.format(primary_key))

            parents = metadata.get_parents(table)
            for parent in parents:
                for foreign_key in metadata.get_foreign_keys(parent, table):
                    extras.append('Foreign key ({}): {}'.format(parent, foreign_key))

            path = metadata.get_table_meta(table).get('path')
            if path is not None:
                extras.append('Data path: {}'.format(path))

            extras = r'\l'.join(extras)

            # Add table node
            title = r'{{{}|{}\l|{}\l}}'.format(table, fields, extras)
        else:
            # Add table node only
            title = r'{{{}}}'.format(table)

        digraph.node(table, label=title)


def _add_edges(metadata, digraph, names, details):
    """Add edges into a `graphviz.Digraph`.

    Each edge represents a relationship between two metadata tables.

    Args:
        digraph (graphviz.Digraph)
    """
    for table in metadata.get_tables():
        for parent in list(metadata.get_parents(table)):
            if names and details:
                label = '\n'.join([
                    '   {}.{} > {}.{}'.format(
                        table, foreign_key,
                        parent, metadata.get_primary_key(parent)
                    )
                    for foreign_key in metadata.get_foreign_keys(parent, table)
                ])
                digraph.edge(
                    parent,
                    table,
                    label=label,
                    arrowhead='oinv'
                )
            else:
                for foreign_key in metadata.get_foreign_keys(parent, table):
                    digraph.edge(parent, table, arrowhead='oinv')


def visualize(metadata, path=None, names=True, details=True):
    """Plot metadata usign graphviz.

    Try to generate a plot using graphviz.
    If a ``path`` is provided save the output into a file.

    Args:
        metadata (Metadata):
            Metadata object to plot.
        path (str):
            Output file path to save the plot, it requires a graphviz
            supported extension. If ``None`` do not save the plot.
            Defaults to ``None``.
    """
    filename, graphviz_extension = _get_graphviz_extension(path)
    digraph = graphviz.Digraph(
        'Metadata',
        format=graphviz_extension,
        node_attr={
            "shape": "Mrecord",
            "fillcolor": "lightgoldenrod1",
            "style": "filled"
        },
    )

    _add_nodes(metadata, digraph, names, details)
    _add_edges(metadata, digraph, names, details)

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
