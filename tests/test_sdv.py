from unittest import TestCase
from unittest.mock import Mock, patch

import pytest

from sdv.sdv import DEFAULT_MODEL, DEFAULT_MODEL_KWARGS, SDV, NotFittedError


def test_save_and_load(tmp_path):
    """Test save and load a SDV instance"""
    # Setup
    metadata = tmp_path / 'save.pkl'
    sdv = SDV()

    # Run "save"
    SDV.save(sdv, str(metadata))

    # Asserts "save"
    assert metadata.exists()
    assert metadata.is_file()

    # Run "load"
    instance = SDV.load(str(metadata))

    # Asserts "load"
    assert isinstance(instance, SDV)

    assert sdv.sampler == instance.sampler
    assert sdv.model == instance.model
    assert sdv.model_kwargs == instance.model_kwargs


class TestSDV(TestCase):

    def test____init__default(self):
        """Create default instance"""
        # Run
        sdv = SDV()

        # Asserts
        assert sdv.model == DEFAULT_MODEL
        assert sdv.model_kwargs == DEFAULT_MODEL_KWARGS
        assert sdv.model_kwargs is not DEFAULT_MODEL_KWARGS

    def test____init__users_params(self):
        """Create default instance"""
        # Run
        sdv = SDV(model='test', model_kwargs={'a': 2})

        # Asserts
        assert sdv.model == 'test'
        assert sdv.model_kwargs == {'a': 2}

    def test__validate_dataset_structure_no_error(self):
        """Test that any error is raised with a supported structure"""
        # Setup
        table_names = ['foo', 'bar', 'tar']
        parents = [[], ['foo'], ['bar']]

        # Run
        sdv = Mock()
        sdv.metadata.get_table_names.return_value = table_names
        sdv.metadata.get_parents.side_effect = parents

        SDV._validate_dataset_structure(sdv)

        # Asserts
        expect_get_parents_call_count = 3
        assert sdv.metadata.get_parents.call_count == expect_get_parents_call_count

    def test__validate_dataset_structure_raise_error(self):
        """Test that a ValueError is raised because the bad structure"""
        # Setup
        table_names = ['foo', 'bar', 'tar']
        parents = [[], [], ['foo', 'bar']]

        # Run & assert
        sdv = Mock()
        sdv.metadata.get_table_names.return_value = table_names
        sdv.metadata.get_parents.side_effect = parents

        with pytest.raises(ValueError):
            SDV._validate_dataset_structure(sdv)

    @patch('sdv.sdv.Sampler')
    @patch('sdv.sdv.Modeler')
    @patch('sdv.sdv.Metadata')
    def test_fit(self, metadata_mock, modeler_mock, sampler_mock):
        """Test fit without the root_path argument"""
        sdv_mock = Mock()

        # Run
        SDV.fit(sdv_mock, '/path/to/metadata', 'test/path')

        # Asserts
        metadata_mock.assert_called_once_with('/path/to/metadata', 'test/path')
        modeler_mock.assert_called_once_with(
            metadata_mock.return_value,
            sdv_mock.model,
            sdv_mock.model_kwargs
        )
        modeler_mock.return_value.model_database.assert_called_once_with()
        sampler_mock.assert_called_once_with(
            metadata_mock.return_value,
            modeler_mock.return_value.models
        )

    def test_sample_fitted(self):
        """Check that the sample is called."""
        # Run
        sdv = Mock()
        table_name = 'DEMO'
        num_rows = 5
        sdv.sampler.sample.return_value = 'test'

        result = SDV.sample(sdv, table_name, num_rows)

        # Asserts
        sdv.sampler.sample.assert_called_once_with(
            'DEMO', 5, sample_children=True, reset_primary_keys=False)

        assert result == 'test'

    def test_sample_not_fitted(self):
        """Check that the sample raise an exception when is not fitted."""
        # Run and asserts
        sdv = Mock()
        sdv.sampler = None
        table_name = 'DEMO'
        num_rows = 5

        with pytest.raises(NotFittedError):
            SDV.sample(sdv, table_name, num_rows)

    def test_sample_all_fitted(self):
        """Check that the sample_all is called"""
        # Run
        sdv = Mock()
        sdv.sampler.sample_all.return_value = 'test'

        result = SDV.sample_all(sdv)

        # Asserts
        sdv.sampler.sample_all.assert_called_once_with(5, reset_primary_keys=False)
        assert result == 'test'

    def test_sample_all_not_fitted(self):
        """Check that the sample_all raise an exception when is not fitted."""
        # Run & asserts
        sdv = Mock()
        sdv.sampler = None

        with pytest.raises(NotFittedError):
            SDV.sample_all(sdv)
