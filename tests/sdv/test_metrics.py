from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy as sp

from sdv.metrics import (
    get_metric_values, mse, r2_score, rmse, score_categorical_coverage, score_stats_dataset,
    score_stats_table)


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


class TestScoreStatsDataset(TestCase):

    def test_raises_error(self):
        """If the table names in both datasets are not equal, an error is raised."""
        # Setup
        real = {
            'a': None,
            'b': None
        }
        synth = {
            'a': None,
            'x': None
        }
        metrics = []
        scores = []
        expected_error_message = "real and synthetic dataset must have the same tables"

        try:
            # Run
            score_stats_dataset(real, synth, metrics=metrics, scores=scores)
        except AssertionError as error:
            # Check
            assert error.args[0] == expected_error_message

    @patch('sdv.metrics.score_stats_table', autospec=True)
    def test_default_call(self, score_table_mock):
        # Setup

        def score_side_effect(*args, **kwargs):
            return 'score_for_table_{}'.format(args[0][-1])

        score_table_mock.side_effect = score_side_effect

        real = {
            'table_A': 'real_data_for_table_A',
            'table_B': 'real_data_for_table_B'
        }
        synth = {
            'table_A': 'synth_data_for_table_A',
            'table_B': 'synth_data_for_table_B'
        }
        metrics = ['metric_1', 'metric_2']
        scores = ['score_1', 'score_2']

        expected_result = {
            'table_A': 'score_for_table_A',
            'table_B': 'score_for_table_B'
        }
        expected_kwargs = dict(
            metrics=['metric_1', 'metric_2'],
            scores=['score_1', 'score_2']
        )
        # Run
        result = score_stats_dataset(real, synth, metrics=metrics, scores=scores)

        # Check
        assert result == expected_result

        score_table_mock.call_args_list == [
            (('real_data_for_table_A', 'synth_data_for_table_A'), expected_kwargs),
            (('real_data_for_table_B', 'synth_data_for_table_B'), expected_kwargs)
        ]


class TestScoreCategoricalCoverage(TestCase):

    def test_same_values(self):
        """If the same values are present on both tables the score is 1."""
        # Setup
        table = pd.DataFrame({
            'A': list('ABCDE'),
            'B': list('ZYXWT')
        })
        categorical_columns = ['A', 'B']

        # Run
        # Note that we pass the same table twice, that is, two identical tables.
        result = score_categorical_coverage(table, table, categorical_columns)

        # Check
        assert result == 1

    def test_raises_error(self):
        """If one of the tables is empty an exception is raised."""
        # Setup
        real = pd.DataFrame()
        synth = pd.DataFrame({
            'A': list('ABCDE'),
            'B': list('ZYXWT')
        })
        categorical_columns = ['A', 'B']

        expected_message = "Can't score empty tables."

        try:
            # Run
            score_categorical_coverage(real, synth, categorical_columns)
        except ValueError as error:
            # Check
            assert error.args[0] == expected_message
