import unittest
from operator import itemgetter
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from sdv.modeler import Modeler
from sdv.sampler import Sampler


class TestSampler(TestCase):

    def test___init__default(self):
        """Test create a default instance of Sampler class"""
        # Run
        sampler = Sampler(None, None)

        # Asserts
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()

    def test__square_matrix(self):
        """Test fill zeros a triangular matrix"""
        # Run
        matrix = [[0.1, 0.5], [0.3]]

        result = Sampler._square_matrix(matrix)

        # Asserts
        expect = [[0.1, 0.5], [0.3, 0.0]]

        assert result == expect

    def test__prepare_sampler_covariance(self):
        """Test prepare_sampler_covariante"""
        # Run
        sampler = Mock()
        sampler._square_matrix.return_value = [[0.4, 0.1], [0.1, 0.0]]
        covariance = [[0.4, 0.1], [0.1]]

        result = Sampler._prepare_sampled_covariance(sampler, covariance)

        # Asserts
        expect = np.array([[0.4, 0.2], [0.2, 0.0]])

        np.testing.assert_equal(result, expect)

    @patch('exrex.getone')
    def test__fill_test_columns(self, mock_exrex):
        """Test fill text columns"""
        # Setup
        mock_exrex.side_effect = ['fake id 1', 'fake id 2']

        metadata_field_meta = [
            {'type': 'id', 'ref': {'table': 'ref_table', 'field': 'ref_field'}},
            {'type': 'id', 'regex': '^[0-9]{10}$'},
            {'type': 'text', 'regex': '^[0-9]{10}$'}
        ]

        sample_rows = {'ref_field': 'some value'}

        # Run
        sampler = Mock()
        sampler.metadata.get_field_meta.side_effect = metadata_field_meta
        sampler.sample_rows.return_value = sample_rows

        data = pd.DataFrame({'tar': ['a', 'b', 'c']})
        columns = ['foo', 'bar', 'tar']
        table_name = 'test'

        result = Sampler._fill_text_columns(sampler, data, columns, table_name)

        # Asserts
        expect = pd.DataFrame({
            'foo': ['some value', 'some value', 'some value'],
            'bar': ['fake id 1', 'fake id 1', 'fake id 1'],
            'tar': ['fake id 2', 'fake id 2', 'fake id 2'],
        })

        pd.testing.assert_frame_equal(result.sort_index(axis=1), expect.sort_index(axis=1))

    def test__reset_primary_keys_generators(self):
        """Test reset values"""
        # Run
        sampler = Mock()
        sampler.primary_key = 'something'
        sampler.remaining_primary_key = 'else'

        Sampler._reset_primary_keys_generators(sampler)

        # Asserts
        assert sampler.primary_key == dict()
        assert sampler.remaining_primary_key == dict()

    def test__transform_synthesized_rows(self):
        """Test transform synthesized rows"""
        # Setup
        metadata_field_names = ['foo', 'bar']
        metadata_reverse_transform = pd.DataFrame({'foo': [0, 1], 'bar': [2, 3], 'tar': [4, 5]})

        # Run
        sampler = Mock()
        sampler.metadata.get_field_names.return_value = metadata_field_names
        sampler.metadata.reverse_transform.return_value = metadata_reverse_transform
        synthesized = None
        table_name = 'test'

        result = Sampler._transform_synthesized_rows(sampler, synthesized, table_name)

        # Asserts
        expect = pd.DataFrame({'foo': [0, 1], 'bar': [2, 3]})

        pd.testing.assert_frame_equal(result, expect)
