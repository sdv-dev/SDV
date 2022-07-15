"""Test Single Table Metadata."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata.single_table import SingleTableMetadata


class TestSingleTableMetadata:
    """Test ``SingleTableMetadata`` class."""

    def test___init__(self):
        """Test creating an instance of ``SingleTableMetadata``."""
        # Run
        instance = SingleTableMetadata()

        # Assert
        assert instance._columns == {}
        assert instance._primary_key is None
        assert instance._alternate_keys == []
        assert instance._constraints == []
        assert instance._version == 'SINGLE_TABLE_V1'
        assert instance._metadata == {
            'columns': {},
            'primary_key': None,
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

    def test_detect_from_dataframe_raises_value_error(self):
        """Test the ``detect_from_dataframe`` method.

        Test that if there are existing columns in the metadata, this raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance._columns``.

        Side Effects:
            Raises a ``ValueError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(ValueError, match=err_msg):
            instance.detect_from_dataframe('dataframe')

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_dataframe(self, mock_print):
        """Test the ``dectect_from_dataframe`` method.

        Test that when given a ``pandas.DataFrame``, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column in the ``pandas.DataFrame``.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - ``pandas.DataFrame`` with multiple data types.

        Side Effects:
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        instance.detect_from_dataframe(data)

        # Assert
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

    def test_detect_from_csv_raises_value_error(self):
        """Test the ``detect_from_csv`` method.

        Test that if there are existing columns in the metadata, this raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Add some value to ``instance._columns``.

        Side Effects:
            Raises a ``ValueError`` stating that ``metadata`` already exists.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns = {'column': {'sdtype': 'categorical'}}

        # Run / Assert
        err_msg = (
            'Metadata already exists. Create a new ``SingleTableMetadata`` '
            'object to detect from other data sources.'
        )

        with pytest.raises(ValueError, match=err_msg):
            instance.detect_from_csv('filepath')

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_csv(self, mock_print):
        """Test the ``dectect_from_csv`` method.

        Test that when given a file path to a ``csv`` file, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column from the read data that is contained within the ``pandas.DataFrame`` from
        that ``csv`` file.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - String that represents the ``path`` to the ``csv`` file.

        Side Effects:
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        with TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'mydata.csv'
            data.to_csv(filepath, index=False)
            instance.detect_from_csv(filepath)

        # Assert
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'categorical'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

    @patch('sdv.metadata.single_table.print')
    def test_detect_from_csv_with_kwargs(self, mock_print):
        """Test the ``dectect_from_csv`` method.

        Test that when given a file path to a ``csv`` file, the current instance of
        ``SingleTableMetadata`` is being updated with the ``sdtypes`` of each
        column from the read data that is contained within the ``pandas.DataFrame`` from
        that ``csv`` file, having in consideration the ``kwargs`` that are passed.

        Setup:
            - Instance of ``SingleTableMetadata``.

        Input:
            - String that represents the ``path`` to the ``csv`` file.

        Side Effects:
            - ``instance._columns`` has been updated with the expected ``sdtypes``.
            - one of the columns must be datetime
            - A message is being printed.
        """
        # Setup
        instance = SingleTableMetadata()
        data = pd.DataFrame({
            'categorical': ['cat', 'dog', 'tiger', np.nan],
            'date': pd.to_datetime(['2021-02-02', np.nan, '2021-03-05', '2022-12-09']),
            'int': [1, 2, 3, 4],
            'float': [1., 2., 3., 4],
            'bool': [np.nan, True, False, True]
        })

        # Run
        with TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'mydata.csv'
            data.to_csv(filepath, index=False)
            instance.detect_from_csv(filepath, pandas_kwargs={'parse_dates': ['date']})

        # Assert
        assert instance._columns == {
            'categorical': {'sdtype': 'categorical'},
            'date': {'sdtype': 'datetime'},
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'}
        }

        expected_print_calls = [
            call('Detected metadata:'),
            call(json.dumps(instance.to_dict(), indent=4))
        ]
        assert mock_print.call_args_list == expected_print_calls

    def test_to_dict(self):
        """Test the ``to_dict`` method from ``SingleTableMetadata``.

        Setup:
            - Instance of ``SingleTableMetadata`` and modify the ``instance._columns`` to ensure
            that ``to_dict`` works properly.
        Output:
            - A dictionary representation of the ``instance`` that does not modify the
              internal dictionaries.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns['my_column'] = 'value'

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        result['columns']['my_column'] = 1
        assert instance._columns['my_column'] == 'value'

    def test__set_metadata_dict(self):
        """Test the ``_set_metadata_dict`` to a instance.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - Dictionary representing ``SingleTableMetadata``.

        Output:
            - ``SingleTableMetadata`` instance with the dictionary represented values.
        """
        # Setup
        instance = SingleTableMetadata()
        metadata = {
            'columns': {'my_column': 'value'},
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance._set_metadata_dict(metadata)

        # Assert
        assert instance._metadata == {
            'columns': {'my_column': 'value'},
            'primary_key': None,
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        assert instance._columns == {'my_column': 'value'}

    def test__load_from_dict(self):
        """Test that ``_load_from_dict`` returns a instance with the ``dict`` updated objects."""
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata._load_from_dict(my_metadata)

        # Assert
        assert instance._metadata == my_metadata
        assert instance._columns == {'my_column': 'value'}
        assert instance._primary_key == 'pk'
        assert instance._alternate_keys == []
        assert instance._constraints == []

    @patch('sdv.metadata.single_table.Path')
    def test_save_to_json_file_exists(self, mock_path):
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'

        # Run
        error_msg = (
            "A file named 'filepath.json' already exists in this folder. Please specify "
            'a different filename.'
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.save_to_json('filepath.json')

    def test_save_to_json(self, mock_path):
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'

        # Run
        error_msg = (
            "A file named 'filepath.json' already exists in this folder. Please specify "
            'a different filename.'
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.save_to_json('filepath.json')

    @patch('sdv.metadata.single_table.json')
    def test___repr__(self, mock_json):
        """Test that the ``__repr__`` method.

        Test that the ``__repr__`` method calls the ``json.dumps``  method and
        returns its output.

        Setup:
            - Instance of ``SingleTableMetadata``.
        Mock:
            - ``json`` from ``sdv.metadata.single_table``.

        Output:
            - ``json.dumps`` return value.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        res = instance.__repr__()

        # Assert
        mock_json.dumps.assert_called_once_with(instance.to_dict(), indent=4)
        assert res == mock_json.dumps.return_value
