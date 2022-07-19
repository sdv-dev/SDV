"""Test Single Table Metadata."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

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
            - Add constraint Mock and ensure that `to_dict` of the object is being called.
        Output:
            - A dictionary representation of the ``instance`` that does not modify the
              internal dictionaries.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns['my_column'] = 'value'
        constraint = Mock()
        constraint.to_dict.return_value = {'column': 'value', 'scalar': 1}
        dict_constraint = {'column': 'value', 'increment_value': 20}
        instance._constraints.extend([constraint, dict_constraint])

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'constraints': [
                {'column': 'value', 'scalar': 1},
                {'column': 'value', 'increment_value': 20}
            ],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        constraint.to_dict.assert_called_once()

        # Ensure that the output object does not alterate the inside object
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
    def test_load_from_json_path_does_not_exist(self, mock_path):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified path does not exist.

        Mock:
            - Mock the ``Path`` library in order to return ``False``, that the file does not exist.

        Input:
            - String representing a filepath.

        Side Effects:
            - A ``ValueError`` is raised pointing that the ``file`` does not exist.
        """
        # Setup
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' does not exist. Please specify a different filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            SingleTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.single_table.open')
    @patch('sdv.metadata.single_table.Path')
    @patch('sdv.metadata.single_table.json')
    def test_load_from_json_schema_not_present(self, mock_json, mock_path, mock_open):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified ``json`` file does
        not contain a ``SCHEMA_VERSION`` in it.

        Mock:
            - Mock the ``Path`` library in order to return ``True``, so the file exists.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.

        Input:
            - String representing a filepath.

        Side Effects:
            - A ``ValueError`` is raised pointing that the given metadata configuration is not
              compatible with the current version.
        """
        # Setup
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'columns': {
                'animals': {
                    'type': 'categorical'
                }
            },
            'primary_key': 'animals',
        }

        # Run / Assert
        error_msg = (
            'This metadata file is incompatible with the ``SingleTableMetadata`` '
            'class and version.'
        )
        with pytest.raises(ValueError, match=error_msg):
            SingleTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.single_table.Constraint')
    @patch('sdv.metadata.single_table.open')
    @patch('sdv.metadata.single_table.Path')
    @patch('sdv.metadata.single_table.json')
    def test_load_from_json(self, mock_json, mock_path, mock_open, mock_constraint):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.
            - Mock the ``Constraint`` to ensure that is being loaded.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
              file (``json.load`` return value)
        """
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_constraint.from_dict.return_value = {'my_constraint': 'my_params'}
        mock_json.load.return_value = {
            'columns': {
                'animals': {
                    'type': 'categorical'
                }
            },
            'primary_key': 'animals',
            'constraints': [{
                'my_constraint': 'my_params'
            }],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata.load_from_json('filepath.json')

        # Assert
        expected_metadata = {
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
            'alternate_keys': [],
            'constraints': [{'my_constraint': 'my_params'}],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }
        assert instance._columns == {'animals': {'type': 'categorical'}}
        assert instance._primary_key == 'animals'
        assert instance._constraints == [{'my_constraint': 'my_params'}]
        assert instance._alternate_keys == []
        assert instance._metadata == expected_metadata
        mock_constraint.from_dict.assert_called_once()

    @patch('sdv.metadata.single_table.Path')
    def test_save_to_json_file_exists(self, mock_path):
        """Test the ``save_to_json`` method.

        Test that when attempting to write over a file that already exists, the method
        raises a ``ValueError``.

        Setup:
            - instance of ``SingleTableMetadata``.
        Mock:
            - Mock ``Path`` in order to point that the file does exist.

        Side Effects:
            - Raise ``ValueError`` pointing that the file does exist.
        """
        # Setup
        instance = SingleTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' already exists in this folder. Please specify "
            'a different filename.'
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.save_to_json('filepath.json')

    def test_save_to_json(self):
        """Test the ``save_to_json`` method.

        Test that ``save_to_json`` stores a ``json`` file and dumps the instance dict into
        it.

        Setup:
            - instance of ``SingleTableMetadata``.
            - Use ``TemporaryDirectory`` to store the file in order to read it afterwards and
              assert it's contents.

        Side Effects:
            - Creates a json representation of the instance.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run / Assert
        with TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir) / 'singletable.json'
            instance.save_to_json(file_name)

            with open(file_name, 'rb') as single_table_file:
                saved_metadata = json.load(single_table_file)
                assert saved_metadata == instance.to_dict()

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
