from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdv.evaluation import (
    DEFAULT_DTYPES, _describe_columns, _describe_table, _get_descriptor_tuples)
from sdv.evaluation.descriptors import DESCRIPTORS, categorical_distribution


class TestDescribeColumns(TestCase):

    def test_single_column(self):
        # Setup
        table = pd.DataFrame({'a': range(10)})
        descriptor = np.mean

        # Run
        result = _describe_columns(table, descriptor)

        # Check
        expected_result = pd.DataFrame({
            'value': [0],
            'statistic': [4.5],
            'descriptor': ['mean'],
            'column': ['a']
        })

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1)
        )

    def test_multiple_columns(self):
        # Setup
        table = pd.DataFrame({
            'a': range(10),
            'b': range(10, 20)
        })
        descriptor = np.mean

        # Run
        result = _describe_columns(table, descriptor)

        # Check
        expected_result = pd.DataFrame({
            'value': [0, 0],
            'statistic': [4.5, 14.5],
            'descriptor': ['mean', 'mean'],
            'column': ['a', 'b']
        })

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1)
        )

    def test_multiple_columns_multiple_outputs(self):
        # Setup
        table = pd.DataFrame({
            'a': list('ABCD'),
            'b': list('WXYZ')
        })
        descriptor = categorical_distribution

        # Run
        result = _describe_columns(table, descriptor)

        # Check
        expected_result = pd.DataFrame({
            'value': ['A', 'B', 'C', 'D', 'W', 'X', 'Y', 'Z'],
            'statistic': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            'descriptor': [
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution',
                'categorical_distribution'
            ],
            'column': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
        })

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1)
        )

    @patch('pandas.concat')
    def test_raise_type_error(self, concat_mock):
        """_describe_columns raise type error"""

        # Setup
        def side_effect_descriptor():
            raise TypeError

        aux = pd.DataFrame()
        concat_mock.return_value = aux

        table = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})

        descriptor = Mock()
        descriptor.__name__ = 'foo'
        descriptor.side_effect = side_effect_descriptor

        # Run
        _describe_columns(table, descriptor)

        # Asserts
        assert descriptor.call_count == 2
        assert concat_mock.call_args == call([], ignore_index=True, sort=False)


class TestDescribeTable(TestCase):

    @patch('sdv.evaluation._describe_columns')
    def test_descriptors(self, descriptor_mock):
        # Setup
        table = pd.DataFrame({'a': range(10)})

        descriptor_mock.return_value = pd.DataFrame({
            'value': [0],
            'statistic': [4.5],
            'descriptor': ['mean'],
            'column': ['a']
        })

        # Run
        table_dtypes = pd.Series({'a': 'int'})
        result = _describe_table(
            table,
            table_dtypes,
            [(np.mean, ('int', 'float')), (categorical_distribution, ('object', 'boolean'))]
        )

        # Asserts
        expected = pd.DataFrame({
            'value': [0],
            'statistic': [4.5],
            'descriptor': ['mean'],
            'column': ['a']
        })

        expected_call_args = pd.DataFrame({
            'a': [True, True, True, True, True, True, True, True, True, True]
        })

        assert result.equals(expected)
        descriptor_mock.call_args[0][0].equals(expected_call_args)


class TestGetDescriptorTuples(TestCase):

    def test_get_descriptor_tuples_instance_str(self):
        descriptors = ['mean', 'std']

        result = _get_descriptor_tuples(descriptors)
        expected = [DESCRIPTORS['mean'], DESCRIPTORS['std']]

        assert result == expected

    def test_get_descriptor_tuples_instance_tuple(self):
        descriptors = [(np.mean, ('int', 'float')), (np.std, ('int', 'float'))]

        result = _get_descriptor_tuples(descriptors)
        expected = [(np.mean, ('int', 'float')), (np.std, ('int', 'float'))]

        assert result == expected

    def test_get_descriptor_tuples_instance_callable(self):
        descriptors = [np.mean]

        result = _get_descriptor_tuples(descriptors)
        expected = [(np.mean, (DEFAULT_DTYPES))]

        assert result == expected
