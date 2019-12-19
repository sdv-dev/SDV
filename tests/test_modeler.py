from unittest import TestCase
from unittest.mock import Mock, call

import pandas as pd

from sdv.metadata import Metadata
from sdv.modeler import Modeler
from sdv.models.base import SDVModel
from sdv.models.copulas import GaussianCopula


class TestModeler(TestCase):

    def test___init__default(self):
        """Test create new Modeler instance with default values"""
        # Run
        modeler = Modeler('test')

        # Asserts
        assert modeler.models == dict()
        assert modeler.metadata == 'test'
        assert modeler.model == GaussianCopula
        assert modeler.model_kwargs == dict()

    def test___init__with_arguments(self):
        # Run
        model = Mock()
        modeler = Modeler({'some': 'metadata'}, model=model, model_kwargs={'some': 'kwargs'})

        # Asserts
        assert modeler.models == dict()
        assert modeler.metadata == {'some': 'metadata'}
        assert modeler.model == model
        assert modeler.model_kwargs == {'some': 'kwargs'}

    def test__get_extensions(self):
        """Test get list of extensions from childs"""
        # Setup
        model = Mock(spec=SDVModel)
        model.return_value = model
        model.get_parameters.side_effect = [
            {'model': 'data 1'},
            {'model': 'data 2'},
            {'model': 'data 3'}
        ]

        modeler = Mock(spec=Modeler)
        modeler.model = model
        modeler.model_kwargs = dict()

        # Run
        child_table = pd.DataFrame({'foo': ['aaa', 'bbb', 'ccc']})
        result = Modeler._get_extension(modeler, 'some_name', child_table, 'foo')

        # Asserts
        expected = pd.DataFrame({
            '__some_name__model': ['data 1', 'data 2', 'data 3'],
            '__some_name__child_rows': [1, 1, 1]
        }, index=['aaa', 'bbb', 'ccc'])
        pd.testing.assert_frame_equal(result, expected)
        assert model.get_parameters.call_count == 3

    def test_cpa_with_tables_no_primary_key(self):
        """Test CPA with tables and no primary key."""
        # Setup
        modeler = Mock(spec=Modeler)
        modeler.metadata = Mock(spec=Metadata)
        modeler.model = Mock(spec=SDVModel)
        modeler.model_kwargs = dict()
        modeler.models = dict()
        modeler.metadata.transform.return_value = pd.DataFrame({'data': [1, 2, 3]})
        modeler.metadata.get_primary_key.return_value = None

        # Run
        tables = {'test': pd.DataFrame({'data': ['a', 'b', 'c']})}
        result = Modeler.cpa(modeler, 'test', tables)

        # Asserts
        expected = pd.DataFrame({'data': [1, 2, 3]})
        expected_transform_call = pd.DataFrame({'data': ['a', 'b', 'c']})

        assert modeler.metadata.load_table.call_count == 0
        assert modeler.metadata.transform.call_args[0][0] == 'test'
        pd.testing.assert_frame_equal(
            modeler.metadata.transform.call_args[0][1],
            expected_transform_call
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_model_database(self):
        """Test model using RCPA"""
        # Setup
        def rcpa_side_effect(table_name, tables):
            tables[table_name] = table_name

        metadata_table_names = ['foo', 'bar', 'tar']
        metadata_parents = [None, 'bar_parent', None]

        modeler = Mock()
        modeler.metadata.get_tables.return_value = metadata_table_names
        modeler.metadata.get_parents.side_effect = metadata_parents
        modeler.rcpa.side_effect = rcpa_side_effect
        modeler.models = dict()

        # Run
        Modeler.model_database(modeler)

        # Asserts
        expected_metadata_parents_call_count = 3
        expected_metadata_parents_call = [call('foo'), call('bar'), call('tar')]
        assert modeler.metadata.get_parents.call_count == expected_metadata_parents_call_count
        assert modeler.metadata.get_parents.call_args_list == expected_metadata_parents_call
