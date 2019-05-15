from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy as sp

from sdv.evaluation import (
    get_descriptor_values, score_descriptors_dataset, score_descriptors_table)
from sdv.evaluation.metrics import mse, r2_score, rmse


class TestGetDescriptorValues(TestCase):

    def test_single_column(self):
        # Setup
        real = pd.DataFrame({'a': range(10)})
        synth = pd.DataFrame({'a': range(20, 10, -1)})
        metric = np.mean

        expected_result = pd.DataFrame({
            'a': [4.5, 15.5]
        })

        # Run
        result = get_descriptor_values(real, synth, metric)

        # Check
        assert result.equals(expected_result)

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

        expected_result = pd.DataFrame({
            'a': [4.5, 15.5],
            'b': [14.5, 5.5],
        })

        # Run
        result = get_descriptor_values(real, synth, metric)

        # Check
        assert result.equals(expected_result)

class TestScoreDescriptorsTable(TestCase):

    @patch('sdv.evaluation.get_descriptor_values', autospec=True)
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

        metrics = [mse_mock, rmse_mock, r2_mock]
        columns = ['mse', 'rmse', 'r2_score']
        index = ['mean', 'std', 'skew', 'kurtosis']
        values = [
            ['{}_{}'.format(score, metric[:4]) for score in columns]
            for metric in index
        ]
        expected_result = pd.DataFrame(values, columns=columns, index=index)

        # Run
        result = score_descriptors_table(real, synth, metrics=metrics)

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


class TestScoreDescriptorsDataset(TestCase):

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
        descriptors = []
        expected_error_message = "real and synthetic dataset must have the same tables"

        try:
            # Run
            score_descriptors_dataset(real, synth, metrics=metrics, descriptors=descriptors)
        except AssertionError as error:
            # Check
            assert error.args[0] == expected_error_message

    @patch('sdv.evaluation.score_descriptors_table', autospec=True)
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
        descriptors = ['descriptor_1', 'score_2']

        expected_result = {
            'table_A': 'score_for_table_A',
            'table_B': 'score_for_table_B'
        }
        expected_kwargs = dict(
            metrics=['metric_1', 'metric_2'],
            scores=['score_1', 'score_2']
        )
        # Run
        result = score_descriptors_dataset(real, synth, metrics=metrics, descriptors=descriptors)

        # Check
        assert result == expected_result

        score_table_mock.call_args_list == [
            (('real_data_for_table_A', 'synth_data_for_table_A'), expected_kwargs),
            (('real_data_for_table_B', 'synth_data_for_table_B'), expected_kwargs)
        ]
