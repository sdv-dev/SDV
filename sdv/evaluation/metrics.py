import numpy as np


def sum_square_diff(x, y):
    """Compute the sum of the square differences of two vectors.

    Args:
        x(np.ndarray): First vector.
        y(np.ndarray): Second vector.

    Returns:
        float: Sum of the squared difference.

    """
    return((x - y)**2).sum()


def r2_score(expected, observed):
    """Compute R2 score.

    Args:
        expected(numpy.ndarray): Ground truth.
        observed(numpy.ndaraay): Observed values.

    Returns:
        float: R2 score.

    """
    numerator = sum_square_diff(expected, observed)
    denominator = sum_square_diff(expected, expected.mean())

    return 1 - (numerator / denominator)


def mse(expected, observed):
    """Compute the Mean Squared Error.

    Args:
        expected(numpy.ndarray): Ground truth.
        observed(numpy.ndaraay): Observed values.

    Returns:
        float: Mean squared error.

    """
    return np.average((expected - observed) ** 2, axis=0)


def rmse(expected, observed):
    """Compute the Root Mean Squared Error.

    Args:
        expected(numpy.ndarray): Ground truth.
        observed(numpy.ndaraay): Observed values.

    Returns:
        float: Root mean squared error.

    """
    return np.sqrt(mse(expected, observed))


DEFAULT_METRICS = (
    mse,
    rmse,
    r2_score
)
