from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy as sp

from sdv.metrics import get_metric_values, mse, r2_score, rmse, score_stats_table


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


class TestScoreStatsTable(TestCase):

    @patch('sdv.metrics.get_metric_values', autospec=True)
    def test_default_call(self, metric_mock):
        # Setup
        real = 'real_data'
        synth = 'synth_data'

        metric_mock.side_effect = [
            ('real_mean', 'synth_mean'),
            ('real_std', 'synth_std'),
            ('real_skew', 'synth_skew'),
            ('real_kurt', 'synth_kurt')
        ]

        def mock_result(name):
            def f(*args):
                return '{}_{}'.format(name, args[0].split('_')[1])

            return f

        mse_mock = MagicMock(spec=mse, side_effect=mock_result('mse'), __name__='mse')
        rmse_mock = MagicMock(spec=rmse, side_effect=mock_result('rmse'), __name__='rmse')
        r2_mock = MagicMock(
            spec=r2_score,
            side_effect=mock_result('r2_score'),
            __name__='r2_score'
        )

        scores = [mse_mock, rmse_mock, r2_mock]
        columns = ['mse', 'rmse', 'r2_score']
        index = ['mean', 'std', 'skew', 'kurtosis']
        values = [
            ['{}_{}'.format(score, metric[:4]) for score in columns]
            for metric in index
        ]
        expected_result = pd.DataFrame(values, columns=columns, index=index)

        # Run
        result = score_stats_table(real, synth, scores=scores)

        # Check
        assert result.equals(expected_result)
        assert metric_mock.call_args_list == [
            (('real_data', 'synth_data', np.mean), {}),
            (('real_data', 'synth_data', np.std), {}),
            (('real_data', 'synth_data', sp.stats.skew), {}),
            (('real_data', 'synth_data', sp.stats.kurtosis), {}),

        ]
        assert mse_mock.call_args_list == [
            (('real_mean', 'synth_mean'), {}),
            (('real_std', 'synth_std'), {}),
            (('real_skew', 'synth_skew'), {}),
            (('real_kurt', 'synth_kurt'), {}),
        ]
        assert rmse_mock.call_args_list == [
            (('real_mean', 'synth_mean'), {}),
            (('real_std', 'synth_std'), {}),
            (('real_skew', 'synth_skew'), {}),
            (('real_kurt', 'synth_kurt'), {}),
        ]
        assert r2_mock.call_args_list == [
            (('real_mean', 'synth_mean'), {}),
            (('real_std', 'synth_std'), {}),
            (('real_skew', 'synth_skew'), {}),
            (('real_kurt', 'synth_kurt'), {}),
        ]
