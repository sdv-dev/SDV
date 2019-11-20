from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import scipy as sp

from sdv.evaluation import evaluate, get_descriptor_values, get_descriptors_table
from sdv.evaluation.descriptors import categorical_distribution


class TestGetDescriptorValues(TestCase):

    def test_single_column(self):
        # Setup
        real = pd.DataFrame({'a': range(10)})
        synth = pd.DataFrame({'a': range(20, 10, -1)})
        descriptor = np.mean

        expected_result = pd.DataFrame({
            'mean_a_0': [4.5, 15.5]
        })

        # Run
        result = get_descriptor_values(real, synth, descriptor)

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
        descriptor = np.mean

        expected_result = pd.DataFrame({
            'mean_a_0': [4.5, 15.5],
            'mean_b_0': [14.5, 5.5],
        })

        # Run
        result = get_descriptor_values(real, synth, descriptor)

        # Check
        assert result.equals(expected_result)

    def test_multiple_columns_multiple_outputs(self):
        # Setup
        real = pd.DataFrame({
            'a': list('ABCD'),
            'b': list('WXYZ')
        })
        synth = pd.DataFrame({
            'a': list('ACDE'),
            'b': list('WUXY')
        })
        descriptor = categorical_distribution

        expected_result = pd.DataFrame([
            {
                'categorical_distribution_a_A': 0.25,
                'categorical_distribution_a_B': 0.25,
                'categorical_distribution_a_C': 0.25,
                'categorical_distribution_a_D': 0.25,
                'categorical_distribution_a_E': np.nan,
                'categorical_distribution_b_U': np.nan,
                'categorical_distribution_b_W': 0.25,
                'categorical_distribution_b_X': 0.25,
                'categorical_distribution_b_Y': 0.25,
                'categorical_distribution_b_Z': 0.25
            },
            {
                'categorical_distribution_a_A': 0.25,
                'categorical_distribution_a_B': np.nan,
                'categorical_distribution_a_C': 0.25,
                'categorical_distribution_a_D': 0.25,
                'categorical_distribution_a_E': 0.25,
                'categorical_distribution_b_U': 0.25,
                'categorical_distribution_b_W': 0.25,
                'categorical_distribution_b_X': 0.25,
                'categorical_distribution_b_Y': 0.25,
                'categorical_distribution_b_Z': np.nan
            }
        ])
        # Run
        result = get_descriptor_values(real, synth, descriptor)

        # Check
        assert result.equals(expected_result)

    @patch('pandas.concat')
    def test_raise_type_error(self, concat_mock):
        """get_descriptor_values raise type error"""

        # Setup
        def side_effect_descriptor():
            raise TypeError

        aux = Mock()
        aux.T = None
        concat_mock.return_value = aux

        # Run
        real = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
        synth = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})

        descriptor = Mock()
        descriptor.__name__ = 'foo'
        descriptor.side_effect = side_effect_descriptor

        get_descriptor_values(real, synth, descriptor)

        # Asserts
        assert descriptor.call_count == 2
        assert concat_mock.call_count == 1
        pd.testing.assert_frame_equal(concat_mock.call_args[0][0][0], pd.DataFrame())


class TestGetDescriptorsTable(TestCase):

    @patch('sdv.evaluation.get_descriptor_values', autospec=True)
    def test_default_call(self, descriptor_mock):
        # Setup
        real = 'real_data'
        synth = 'synth_data'

        descriptor_mock.side_effect = [
            pd.DataFrame({'a': [0, 1], 'b': [1, 0]}),
            pd.DataFrame({'x': [0, 1], 'y': [1, 0]}),
            pd.DataFrame({'c': [0, 1], 'd': [1, 0]}),
            pd.DataFrame({'u': [0, 1], 'v': [1, 0]}),
            pd.DataFrame({'r': [0, 1], 's': [1, 0]})
        ]

        expected_result = pd.DataFrame({
            'a': [0, 1],
            'b': [1, 0],
            'x': [0, 1],
            'y': [1, 0],
            'c': [0, 1],
            'd': [1, 0],
            'u': [0, 1],
            'v': [1, 0],
            'r': [0, 1],
            's': [1, 0]
        }, columns=list('abxycduvrs'))

        # Run
        result = get_descriptors_table(real, synth)

        # Check
        assert result.equals(expected_result)
        assert descriptor_mock.call_count == 5
        descriptor_mock.assert_has_calls([
            call('real_data', 'synth_data', np.mean),
            call('real_data', 'synth_data', np.std),
            call('real_data', 'synth_data', sp.stats.skew),
            call('real_data', 'synth_data', sp.stats.kurtosis),
            call('real_data', 'synth_data', categorical_distribution)
        ], any_order=True)

    @patch('sdv.evaluation.DESCRIPTORS')
    @patch('sdv.evaluation.get_descriptor_values', autospec=True)
    @patch('sdv.evaluation.pd.concat', autospec=True)
    def test_string_descriptor(self, concat_mock, get_descriptor_mock, descriptors_mock):
        """If a descriptor is a string it will changed by its value in DESCRIPTORS."""
        # Setup
        real = 'real data'
        synth = 'synth data'
        descriptors = [
            'a_descriptor_string',
        ]
        expected_result = 'None'

        concat_mock.return_value = 'concatenated descriptors'
        get_descriptor_mock.return_value = 'descriptor values'
        descriptor_value = MagicMock(__name__='descriptor', return_value='a_descriptor_function')
        descriptors_mock.__getitem__ = descriptor_value
        expected_result = 'concatenated descriptors'

        # Run
        result = get_descriptors_table(real, synth, descriptors=descriptors)

        # Check
        assert result == expected_result
        descriptors_mock.__getitem__.assert_called_once_with('a_descriptor_string')
        get_descriptor_mock.assert_called_once_with(
            'real data', 'synth data', 'a_descriptor_function')


class TestEvaluate(TestCase):

    def test_raises_error(self):
        """If the table names in both datasets are not equal, an error is raised."""
        # Setup
        metadata = dict()

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
            evaluate(metadata, synth, real=real, metrics=metrics, descriptors=descriptors)
        except AssertionError as error:
            # Check
            assert error.args[0] == expected_error_message

    @patch('sdv.evaluation.get_descriptors_table', autospec=True)
    def test_single_table(self, descriptors_mock):
        # Setup
        descriptors_mock.return_value = pd.DataFrame([
            {
                'a': 1,
                'b': 2,
                'c': 3
            },
            {
                'a': 2,
                'b': 4,
                'c': 6
            },

        ])

        metadata = {
            'fields': dict()
        }

        real = pd.DataFrame({'a': [1, 0]})
        synth = pd.DataFrame({'a': [0, 1]})

        metric_1 = MagicMock(return_value=0, __name__='metric_1')
        metric_2 = MagicMock(return_value=1, __name__='metric_2')

        metrics = [metric_1, metric_2]
        descriptors = ['descriptor_1', 'descriptors_2']

        expected_result = pd.Series({
            'metric_1': 0,
            'metric_2': 1
        })

        # Run
        result = evaluate(metadata, synth, real=real, metrics=metrics, descriptors=descriptors)

        # Check
        assert result.equals(expected_result)
        descriptors_mock.assert_called_once_with(real, synth, descriptors)

        call_args_list = metric_1.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        assert kwargs == {}
        assert len(args) == 2
        assert args[0].equals(pd.Series({'a': 1, 'b': 2, 'c': 3}, name=0))
        assert args[1].equals(pd.Series({'a': 2, 'b': 4, 'c': 6}, name=1))

        call_args_list = metric_1.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        assert kwargs == {}
        assert len(args) == 2
        assert args[0].equals(pd.Series({'a': 1, 'b': 2, 'c': 3}, name=0))
        assert args[1].equals(pd.Series({'a': 2, 'b': 4, 'c': 6}, name=1))

    @patch('pandas.DataFrame.drop')
    @patch('sdv.evaluation.get_descriptors_table')
    def test_evaluate_dict_instance(self, descriptors_table_mock, pandas_drop_mock):
        """evaluate with dict instances"""

        # Setup
        descriptors_table_mock.return_value = pd.DataFrame({
            'foo': [1, 0]
        })

        # Run

        metadata = {
            'tables': [
                {
                    'name': 'a',
                    'fields': [
                        {
                            'name': 'a_pk_field',
                            'type': 'id'
                        }, {
                            'name': 'a_field',
                            'type': 'numerical',
                            'subtype': 'integer'
                        }
                    ]
                }, {
                    'name': 'b',
                    'fields': [
                        {
                            'name': 'b_fk_field',
                            'ref': {'field': 'a_pk_field', 'table': 'a'},
                            'type': 'id'
                        }, {
                            'name': 'b_field',
                            'type': 'numerical',
                            'subtype': 'float'
                        }
                    ]
                }
            ]
        }

        real = {
            'a': pd.DataFrame({
                'a_pk_field': [1, 2, 3, 4],
                'a_field': [10, 14, 11, 9]
            }),
            'b': pd.DataFrame({
                'b_fk_field': [1, 2, 3, 4],
                'b_field': [3.4, 1.1, 2.5, 4.1]
            })
        }

        synth = {
            'a': pd.DataFrame({
                'a_pk_field': [1, 2, 3, 4],
                'a_field': [10, 14, 11, 9]
            }),
            'b': pd.DataFrame({
                'b_fk_field': [1, 2, 3, 4],
                'b_field': [3.4, 1.1, 2.5, 4.1]
            })
        }

        result = evaluate(metadata, synth, real)

        # Asserts
        expected_drop_ids = [
            call(['a_pk_field'], axis=1, inplace=True),
            call(['a_pk_field'], axis=1, inplace=True),
            call(['b_fk_field'], axis=1, inplace=True),
            call(['b_fk_field'], axis=1, inplace=True)
        ]

        assert descriptors_table_mock.call_count == 2

        for res, exp in zip(sorted(pandas_drop_mock.call_args_list), sorted(expected_drop_ids)):
            assert res == exp

        pd.testing.assert_series_equal(result, pd.Series({
            'mse': 1.0,
            'rmse': 1.0,
            'r2_score': -float("Inf")
        }))
