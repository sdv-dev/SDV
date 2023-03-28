from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdv.data_processing.numerical_formatter import NumericalFormatter


class TestNumericalFormatter:

    def test___init__(self):
        """Test ``__init__`` attributes properly set."""
        # Run
        formatter = NumericalFormatter(
            enforce_rounding=True,
            enforce_min_max_values=True,
            computer_representation='Int8'
        )

        # Assert
        assert formatter.enforce_rounding is True
        assert formatter.enforce_min_max_values is True
        assert formatter.computer_representation == 'Int8'

    @patch('sdv.data_processing.numerical_formatter.LOGGER')
    def test__learn_rounding_digits_more_than_15_decimals(self, log_mock):
        """Test the ``_learn_rounding_digits`` method with more than 15 decimals.

        If the data has more than 15 decimals, return None and use ``LOGGER`` to inform the user.
        """
        # Setup
        data = pd.Series(np.random.random(size=10).round(20), name='col')

        # Run
        output = NumericalFormatter._learn_rounding_digits(data)

        # Assert
        log_msg = (
            "No rounding scheme detected for column 'col'. Synthetic data will not be rounded."
        )
        log_mock.info.assert_called_once_with(log_msg)
        assert output is None

    def test__learn_rounding_digits_less_than_15_decimals(self):
        """Test the ``_learn_rounding_digits`` method with less than 15 decimals.

        If the data has less than 15 decimals, the maximum number of decimals should be returned.

        Input:
            - an array that contains floats with a maximum of 3 decimals and a NaN.

        Output:
            - 3
        """
        # Setup
        data = pd.Series(np.array([10, 0., 0.1, 0.12, 0.123, np.nan]))

        # Run
        output = NumericalFormatter._learn_rounding_digits(data)

        # Assert
        assert output == 3

    def test__learn_rounding_digits_negative_decimals_float(self):
        """Test the ``_learn_rounding_digits`` method with floats multiples of powers of 10.

        If the data has all multiples of 10 the output should be None.

        Input:
            - an array that contains floats that are multiples of 10, 100 and 1000 and a NaN.
        """
        # Setup
        data = pd.Series(np.array([1230., 12300., 123000., np.nan]))

        # Run
        output = NumericalFormatter._learn_rounding_digits(data)

        # Assert
        assert output == 0

    def test__learn_rounding_digits_negative_decimals_integer(self):
        """Test the ``_learn_rounding_digits`` method with integers multiples of powers of 10.

        If the data has all multiples of 10 the output should be None.

        Input:
            - an array that contains integers that are multiples of 10, 100 and 1000 and a NaN.
        """
        # Setup
        data = pd.Series(np.array([1230, 12300, 123000, np.nan]))

        # Run
        output = NumericalFormatter._learn_rounding_digits(data)

        # Assert
        assert output == 0

    def test__learn_rounding_digits_all_nans(self):
        """Test the ``_learn_rounding_digits`` method with data that is all NaNs.

        If the data is all NaNs, expect that the output is 0.

        Input:
            - an array of NaN.
        """
        # Setup
        data = pd.Series(np.array([np.nan, np.nan, np.nan, np.nan]))

        # Run
        output = NumericalFormatter._learn_rounding_digits(data)

        # Assert
        assert output is None

    def test_learn_format(self):
        """Test that ``learn_format`` method.

        Ensure attributes are correct when ``enforce_min_max_values`` and
        ``enforce_rounding`` are False.

        Setup:
            - a NumericalFormatter with ``_validate_values_within_bounds`` mocked.

        Input:
            - a pandas series.

        Side Effect:
            - only ``_dtype`` is set.
        """
        # Setup
        data = pd.Series([1.5, None, 2.5])
        formatter = NumericalFormatter(enforce_min_max_values=False, enforce_rounding=False)
        formatter._validate_values_within_bounds = Mock()

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._dtype == float
        assert formatter._min_value is None
        assert formatter._max_value is None
        assert formatter._rounding_digits is None

    def test_learn_format_rounding_scheme_true(self):
        """Test ``learn_format`` with ``enforce_rounding`` set to ``True``.

        If ``enforce_rounding`` is set to ``True``, the ``learn_format`` method
        should set its ``_rounding_digits`` instance variable to what is learned
        in the data.

        Input:
            - a Series with floats up to 4 decimals and a None value

        Side Effect:
            - ``_rounding_digits`` is set to 4
        """
        # Setup
        data = pd.Series([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9, None])
        formatter = NumericalFormatter(enforce_rounding=True)

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._rounding_digits == 4

    def test__fit_enforce_rounding_true_max_decimals(self):
        """Test ``learn_format`` with ``enforce_rounding`` set to ``True``.

        If the ``enforce_rounding`` parameter is set to ``True``, ``learn_format`` should
        learn the ``_rounding_digits`` to be the max number of decimal places seen in the data.
        The max amount of decimals that floats can be accurately compared with is 15.
        If the input data has values with more than 14 decimals, we will not be able to
        accurately learn the number of decimal places required, so we do not round.

        Input:
            - Series with a value that has 15 decimals
        """
        # Setup
        data = pd.Series([0.000000000000001])
        formatter = NumericalFormatter(enforce_rounding=True)

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._rounding_digits is None

    def test_learn_format_enforce_rounding_true_inf(self):
        """Test ``learn_format`` with ``enforce_rounding`` set to ``True``.

        If the ``enforce_rounding`` parameter is set to ``True``, and the data
        contains only integers or infinite values, ``learn_format`` should learn
        ``_rounding_digits`` to be None.


        Input:
            - Series with ``np.inf`` as a value

        Side Effect:
            - ``_rounding_digits`` is set to None
        """
        # Setup
        data = pd.Series([15000, 4000, 60000, np.inf])
        formatter = NumericalFormatter(enforce_rounding=True)

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._rounding_digits == 0

    def test_learn_format_enforce_rounding_true_max_zero(self):
        """Test ``learn_format`` with ``enforce_rounding`` set to ``True``.

        If the ``enforce_rounding`` parameter is set to ``True``, and the max
        in the data is 0, ``learn_format`` should learn the ``_rounding_digits`` to be None.

        Input:
            - Series with 0 as max value

        Side Effect:
            - ``_rounding_digits`` is set to None
        """
        # Setup
        data = pd.Series([0, 0, 0])
        formatter = NumericalFormatter(enforce_rounding=True)

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._rounding_digits == 0

    def test_learn_format_enforce_min_max_values_true(self):
        """Test ``_fit`` with ``enforce_min_max_values`` set to ``True``.

        If the ``enforce_min_max_values`` parameter is set to ``True``,
        the ``learn_format`` method should learn the min and max values from the data.

        Input:
            - Series of floats and null values

        Side Effect:
            - ``_min_value`` and ``_max_value`` are learned
        """
        # Setup
        data = pd.Series([-100, -5000, 0, None, 100, 4000])
        formatter = NumericalFormatter(enforce_min_max_values=True)

        # Run
        formatter.learn_format(data)

        # Asserts
        assert formatter._min_value == -5000
        assert formatter._max_value == 4000

    def test_format_data_enforce_rounding_false(self):
        """Test ``format_data`` when ``enforce_rounding`` is ``False``.

        The data should not be rounded at all.

        Input:
            - random Series of floats between 0 and 1

        Output:
            - input array
        """
        # Setup
        data = pd.Series(np.random.random(10))
        formatter = NumericalFormatter(enforce_rounding=False)
        formatter._rounding_digits = None

        # Run
        result = formatter.format_data(data)

        # Assert
        np.testing.assert_array_equal(result, data)

    def test_format_data_rounding_none_dtype_int(self):
        """Test ``format_data`` with ``_dtype`` as ``np.int64`` and no rounding.

        The data should be rounded to 0 decimals and returned as integer values if the ``_dtype``
        is ``np.int64`` even if ``_rounding_digits`` is ``None``.

        Input:
            - Series of multiple float values with decimals

        Output:
            - input array rounded an converted to integers
        """
        # Setup
        data = pd.Series([0., 1.2, 3.45, 6.789])
        formatter = NumericalFormatter()
        formatter._rounding_digits = None
        formatter._dtype = np.int64

        # Run
        result = formatter.format_data(data)

        # Assert
        expected = np.array([0, 1, 3, 7])
        np.testing.assert_array_equal(result, expected)

    def test_format_data_rounding_small_numbers(self):
        """Test ``format_data`` when ``_rounding_digits`` is positive.

        The data should round to the maximum number of decimal places
        set in the ``_rounding_digits`` value.

        Input:
            - Series with decimals

        Output:
            - same array rounded to the provided number of decimal places
        """
        # Setup
        data = pd.Series([1.1111, 2.2222, 3.3333, 4.44444, 5.555555])
        formatter = NumericalFormatter()
        formatter.enforce_rounding = True
        formatter._rounding_digits = 2

        # Run
        result = formatter.format_data(data)

        # Assert
        expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
        np.testing.assert_array_equal(result, expected_data)

    def test_format_data_rounding_big_numbers_type_int(self):
        """Test ``format_data`` when ``_rounding_digits`` is negative.

        The data should round to the number set in the ``_rounding_digits``
        attribute and remain ints.

        Input:
            - Series with with floats above 100

        Output:
            - same array rounded to the provided number of 0s
            - array should be of type int
        """
        # Setup
        data = pd.Series([2000.0, 120.0, 3100.0, 40100.0])
        formatter = NumericalFormatter()
        formatter._dtype = int
        formatter.enforce_rounding = True
        formatter._rounding_digits = -3

        # Run
        result = formatter.format_data(data)

        # Assert
        expected_data = np.array([2000, 0, 3000, 40000])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == int

    def test_format_data_rounding_negative_type_float(self):
        """Test ``format_data`` when ``_rounding_digits`` is negative.

        The data should round to the number set in the ``_rounding_digits``
        attribute and remain floats.

        Input:
            - Series with with larger numbers

        Output:
            - same array rounded to the provided number of 0s
            - array should be of type float
        """
        # Setup
        data = pd.Series([2000.0, 120.0, 3100.0, 40100.0])
        formatter = NumericalFormatter()
        formatter.enforce_rounding = True
        formatter._rounding_digits = -3

        # Run
        result = formatter.format_data(data)

        # Assert
        expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == float

    def test_format_data_rounding_zero_decimal_places(self):
        """Test ``format_data`` when ``_rounding_digits`` is 0.

        The data should round to the number set in the ``_rounding_digits`` attribute.

        Input:
            - Series with with larger numbers

        Output:
            - same array rounded to the 0s place
        """
        # Setup
        data = pd.Series([2000.554, 120.2, 3101, 4010])
        formatter = NumericalFormatter()
        formatter.enforce_rounding = True
        formatter._rounding_digits = 0

        # Run
        result = formatter.format_data(data)

        # Assert
        expected_data = np.array([2001, 120, 3101, 4010])
        np.testing.assert_array_equal(result, expected_data)

    def test_format_data_enforce_min_max_values(self):
        """Test ``format_data`` with ``enforce_min_max_values`` set to ``True``.

        The ``format_data`` method should clip any values above
        the ``max_value`` and any values below the ``min_value``.

        Input:
            - Series with values above the max and below the min

        Output:
            - array with out of bound values clipped to min and max
        """
        # Setup
        data = pd.Series([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])
        formatter = NumericalFormatter()
        formatter.enforce_min_max_values = True
        formatter._max_value = 400
        formatter._min_value = -300

        # Run
        result = formatter.format_data(data)

        # Asserts
        np.testing.assert_array_equal(result, np.array([-300, -300, -300, -250, 0, 125, 400, 400]))

    def test_format_data_enforce_computer_representation(self):
        """Test ``format_data`` with ``computer_representation`` set to ``Int8``.

        The ``format_data`` method should clip any values out of bounds.

        Input:
            - Series with values above the max and below the min

        Output:
            - array with out of bound values clipped to min and max
        """
        # Setup
        data = pd.Series([-np.inf, np.nan, -5000, -301, -100, 0, 125, 401, np.inf])
        formatter = NumericalFormatter(computer_representation='Int8')

        # Run
        result = formatter.format_data(data)

        # Asserts
        np.testing.assert_array_equal(
            result, np.array([-128, np.nan, -128, -128, -100, 0, 125, 127, 127])
        )
