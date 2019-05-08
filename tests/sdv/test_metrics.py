from unittest import TestCase

import numpy as np
import pandas as pd

from sdv.metrics import get_metric_values


class TestGetMetricValues(TestCase):

    def test_single_column(self):
        # Setup
        real = pd.DataFrame({'a': range(10)})
        synth = pd.DataFrame({'a': range(20, 10, -1)})
        metric = np.mean

        expected_result = (
            np.array([4.5]),
            np.array([15.5])
        )

        # Run
        result = get_metric_values(real, synth, metric)

        # Check
        assert result == expected_result

    def test_multiple_columns(self):
        # Setup
        real = pd.DataFrame({
            'a': range(10),
            'b': range(10, 20)
        })
        synth = pd.DataFrame({
            'a': range(20, 10, -1),
            'b': range(10, 0, -1)
        })
        metric = np.mean

        expected_real = np.array([4.5, 14.5])
        expected_synth = np.array([15.5, 5.5])

        # Run
        result_real, result_synth = get_metric_values(real, synth, metric)

        # Check
        np.testing.assert_equal(result_real, expected_real)
        np.testing.assert_equal(result_synth, expected_synth)
