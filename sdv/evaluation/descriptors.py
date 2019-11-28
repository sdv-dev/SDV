import enum

import numpy as np
import scipy as sp


class DTypes(enum.Enum):
    INT = 'int'
    FLOAT = 'float'
    STR = 'str'
    BOOL = 'bool'
    DATETIME = 'datetime64'

    @staticmethod
    def values():
        return [item.value for item in list(DTypes)]


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
    'mean': (np.mean, (DTypes.INT, DTypes.FLOAT, DTypes.BOOL)),
    'std': (np.std, (DTypes.INT, DTypes.FLOAT, DTypes.BOOL)),
    'skew': (sp.stats.skew, (DTypes.INT, DTypes.FLOAT, DTypes.BOOL)),
    'kurtosis': (sp.stats.kurtosis, (DTypes.INT, DTypes.FLOAT, DTypes.BOOL)),
    'categorical_distribution': (categorical_distribution, (DTypes.STR, DTypes.BOOL))
}
