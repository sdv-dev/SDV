import numpy as np
import scipy as sp


def categorical_distribution(column):
    """Compute the empirical distribution for a categorical column.

    Args:
        column(pandas.Series): Column to compute the empirical distribution.

    Returns:
        pandas.Series: Serie whose index are the catogories, and their relative frequency is
                       their value.

    """
    return column.value_counts(normalize=True).sort_index()


DESCRIPTORS = {
    'mean': (np.mean, ('int', 'float', 'bool')),
    'std': (np.std, ('int', 'float', 'bool')),
    'skew': (sp.stats.skew, ('int', 'float', 'bool')),
    'kurtosis': (sp.stats.kurtosis, ('int', 'float', 'bool')),
    'categorical_distribution': (categorical_distribution, ('object', 'bool'))
}
