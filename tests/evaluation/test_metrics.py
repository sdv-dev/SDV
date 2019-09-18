from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from sdv.evaluation.metrics import mse, r2_score, rmse, sum_square_diff


class TestMetrics(TestCase):

    def test_sum_square_diff(self):
        # Setup
        x = np.zeros(5)
        y = np.array(range(5))
        expected_result = 30

        # Run
        result = sum_square_diff(x, y)

        # Check
        assert result == expected_result

    @patch('sdv.evaluation.metrics.sum_square_diff', autospec=True)
    def test_r2_score(self, sum_mock):
        """If both imputs are identical, r2_score is 1."""
        # Setup
        expected = MagicMock()
        expected.mean.return_value = 'mean of expected'
        observed = 'observed'

        sum_result = sum_mock.return_value
        div_result = sum_result.__truediv__.return_value
        div_result.__rsub__.return_value = 'result'

        # Return
        result = r2_score(expected, observed)

        # Check
        assert result == 'result'

        expected.mean.assert_called_once_with()
        assert sum_mock.call_args_list == [
            ((expected, 'observed'), {}),
            ((expected, 'mean of expected'), {})
        ]
        sum_result.__truediv__.assert_called_once_with(sum_result)
        div_result.__rsub__.assert_called_once_with(1)

    @patch('sdv.evaluation.metrics.np.average', autospec=True)
    def test_mse(self, average_mock):
        # Setup
        expected = MagicMock()
        diff_mock = expected.__sub__.return_value
        diff_mock.__pow__.return_value = 'squared differences'
        observed = 'observed'

        average_mock.return_value = 'average'

        # Run
        result = mse(expected, observed)

        # Check
        assert result == 'average'
        expected.__sub__.assert_called_once_with('observed')
        diff_mock.__pow__.assert_called_once_with(2)
        average_mock.assert_called_once_with('squared differences', axis=0)

    @patch('sdv.evaluation.metrics.np.sqrt', autospec=True)
    @patch('sdv.evaluation.metrics.mse', autospec=True)
    def test_rmse(self, mse_mock, sqrt_mock):
        # Setup
        expected = 'expected'
        observed = 'observed'
        mse_mock.return_value = 'mse value'
        sqrt_mock.return_value = 'rmse'

        # Run
        result = rmse(expected, observed)

        # Check
        assert result == 'rmse'
        mse_mock.assert_called_once_with('expected', 'observed')
        sqrt_mock.assert_called_once_with('mse value')
