from collections import defaultdict

import numpy as np
import pandas as pd


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


DEFAULT_SCORES = (
    mse,
    rmse,
    r2_score
)

DEFAULT_METRICS = (
    np.mean,
    np.std,
    np.stats.skew,
    np.stats.kurt
)


def get_metric_values(real, synth, metric):
    """Compute the metric values for the given tables.

    Args:
        real(pandas.DataFrame): Real data.
        synth(pandas.DataFrame): Synthesized data.
        metric(callable): Callable that accepts columns and returns real-values.

    Return:
        tuple(numpy.ndarray, numpy.ndarray): Result of metric computed column-wise on both tables.

    """
    real_values = real.apply(metric, axis=0).values
    synth_values = synth.apply(metric, axis=0).values

    return real_values, synth_values


def score_table(real, synth, scores=DEFAULT_SCORES, metrics=DEFAULT_METRICS):
    """Score the synthesized data using the given scores and metrics.

    Args:
        real(pandas.DataFrame)
        synth(pandas.DataFrame)
        scores(list(callable))
        metrics(list(callable))

    Return:
        pandas.DataFrame
    
    """
    score_values = defaultdict(list)
    index = []
    for metric in metrics:
        index.append(metric.name)
        real_metric, synth_metric = get_metric_values(real, synth, metric)
        for score in scores:
            score_value = score(real_metric, synth_metric)
            score_values[score.__name__].append(score_value)

    return pd.DataFrame(score_values, index=index)
