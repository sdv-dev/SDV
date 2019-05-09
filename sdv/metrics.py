from collections import defaultdict

import numpy as np
import pandas as pd
import scipy as sp


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
    sp.stats.skew,
    sp.stats.kurtosis
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


def score_stats_table(real, synth, metrics=DEFAULT_METRICS, scores=DEFAULT_SCORES):
    """Score the synthesized data using the given scores and metrics.

    Args:
        real(pandas.DataFrame): Table of real data.
        synth(pandas.DataFrame): Table of synthesized data.
        metrics(list(callable)): List of metrics.
        scores(list(callable)): List of scores.

    Return:
        pandas.DataFrame:
            DataFrame whose columns are the names of the scores, as index the name of the metric,
            and as a values, the value of the score applied to the metric of both tables.

    """
    score_values = defaultdict(list)
    index = []
    columns = [score.__name__ for score in scores]
    for metric in metrics:
        index.append(metric.__name__)
        real_metric, synth_metric = get_metric_values(real, synth, metric)
        for score in scores:
            score_value = score(real_metric, synth_metric)
            score_values[score.__name__].append(score_value)

    return pd.DataFrame(score_values, index=index, columns=columns)


def score_stats_dataset(real, synth, metrics=DEFAULT_METRICS, scores=DEFAULT_SCORES):
    """Compute stats score for all tables.

        Args:
        real(dict[str, pandas.DataFrame]): Map of names and tables of real data.
        synth(dict[str, pandas.DataFrame]): Map of names and tables of synthesized data.
        metrics(list(callable)): List of metrics.
        scores(list(callable)): List of scores.

    Return:
        dict[str, pandas.DataFrame]:
            Dictionary with the table name as keys and as values a DataFrame whose columns are
            the names of the scores, as index the name of the metric, and as a values, the value
            of the score applied to the metric of both tables.

    """
    assert real.keys() == synth.keys(), "real and synthetic dataset must have the same tables"

    result = {}
    for name, real_data in real.items():
        synth_data = synth[name]
        result[name] = score_stats_table(real_data, synth_data, metrics=metrics, scores=scores)

    return result


def score_categorical_coverage(real, synth, categorical_columns):
    """Return the proportion of unique categorical values combination covered by synthetic data.

    Args:
        real(pandas.DataFrame): Table of real data.
        synth(pandas.DataFrame): Table of synthesized data.
        categorical_columns(list[str]): List of labels of categorical columns.

    Returns:
        float: Proportion of u
    """
    if not (real.shape[0] and synth.shape[0]):
        raise ValueError("Can't score empty tables.")

    real_unique = real.drop_duplicates(subset=categorical_columns).shape[0]
    synth_unique = synth.drop_duplicates(subset=categorical_columns).shape[0]

    return synth_unique / real_unique
