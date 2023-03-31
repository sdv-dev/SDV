import pandas as pd

from sdv.data_processing.datetime_formatter import DatetimeFormatter


class TestDatetimeFormatter:

    def test___init__(self):
        """Test ``__init__`` attributes by default."""
        # Run
        formatter = DatetimeFormatter()

        # Assert
        assert formatter.datetime_format is None

    def test___init__with_datetime_format(self):
        """Test ``__init__`` attributes are properly set."""
        # Run
        formatter = DatetimeFormatter(datetime_format='%Y-%m-%d')

        # Assert
        assert formatter.datetime_format == '%Y-%m-%d'

    def test_learn_format(self):
        """Test that ``learn_format`` learns the expected format of the datetime and dtype."""
        # Setup
        formatter = DatetimeFormatter()
        column = pd.Series(['2021-02-15', '2022-05-16', '2023-04-11'])

        # Run
        formatter.learn_format(column)

        # Assert
        assert formatter._dtype == 'O'
        assert formatter.datetime_format == '%Y-%m-%d'

    def test_learn_format_with_set_datetime(self):
        """Test that ``learn_format`` learns only the dtype."""
        # Setup
        formatter = DatetimeFormatter('%m-%d-%Y')
        column = pd.Series(pd.to_datetime(['2021-02-15', '2022-05-16', '2023-04-11']))

        # Run
        formatter.learn_format(column)

        # Assert
        assert formatter._dtype == '<M8[ns]'
        assert formatter.datetime_format == '%m-%d-%Y'

    def test_format_data(self):
        """Test that formats the input data as expected."""
        # Setup
        formatter = DatetimeFormatter('%d-%m-%Y')
        formatter._dtype = 'O'
        column = pd.Series(['2021-02-15', '2022-05-16', '2023-04-11'])

        # Run
        result = formatter.format_data(column)

        # Assert
        pd.testing.assert_series_equal(
            result,
            pd.Series(['15-02-2021', '16-05-2022', '11-04-2023'])
        )

    def test_format_datetime_does_not_match_format(self):
        """Test that datetime column can be formatted if the input doesn't match the format."""
        # Setup
        formatter = DatetimeFormatter('%Y%m%d%H%M%S%f')
        formatter._dtype = 'O'
        column = pd.Series([
            '2 Sep 2022 11:04:43',
            '16 Sep 2022 23:03:56',
            '26 Aug 2022 17:39:17',
            '26 Aug 2022 21:21:35',
            '29 Sep 2022 11:13:11'
        ])

        # Run
        result = formatter.format_data(column)

        # Assert
        expected = pd.Series([
            '20220902110443000000',
            '20220916230356000000',
            '20220826173917000000',
            '20220826212135000000',
            '20220929111311000000'
        ])
        pd.testing.assert_series_equal(result, expected)
