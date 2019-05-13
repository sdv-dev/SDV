import numpy as np
import scipy as sp


def score_categorical_coverage(real, synth, categorical_columns):
    """Return the proportion of unique categorical values combination covered by synthetic data.

    Args:
        real(pandas.DataFrame): Table of real data.
        synth(pandas.DataFrame): Table of synthesized data.
        categorical_columns(list[str]): List of labels of categorical columns.

    Returns:
        float: Proportion of categorical combinations.

    """
    if not (real.shape[0] and synth.shape[0]):
        raise ValueError("Can't score empty tables.")

    real_unique = real.drop_duplicates(subset=categorical_columns).shape[0]
    synth_unique = synth.drop_duplicates(subset=categorical_columns).shape[0]

    return synth_unique / real_unique


DESCRIPTORS = {
    'mean': np.mean,
    'std': np.std,
    'skewness': sp.stats.skew,
    'kurtosis': sp.stats.kurtosis
}

DEFAULT_DESCRIPTORS = list(DESCRIPTORS.values())
