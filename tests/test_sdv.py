from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
from copulas import NotFittedError

from sdv import SDV
from sdv.modeler import DEFAULT_MODEL


def test_save_and_load(tmp_path):
    """Test save and load a SDV instance"""
    # Setup
    metadata = tmp_path / "save.pkl"
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
    assert sdv.distribution == instance.distribution
    assert sdv.model_kwargs == instance.model_kwargs


class TestSDV(TestCase):

    def test____init__default(self):
        """Create default instance"""
        # Run
        sdv = SDV()

        # Asserts
        assert sdv.model == DEFAULT_MODEL
        assert sdv.distribution is None
        assert sdv.model_kwargs is None

    def test____init__users_params(self):
        """Create default instance"""
        # Run
        sdv = SDV(model='test', distribution=1, model_kwargs={'a': 2})

        # Asserts
        assert sdv.model == 'test'
        assert sdv.distribution == 1
        assert sdv.model_kwargs == {'a': 2}

    def test__check_unsupported_dataset_structure_no_error(self):
        """Test that any error is raised with a supported structure"""
        # Setup
        table_names = ['foo', 'bar', 'tar']
        parents = [[], ['foo'], ['bar']]

        # Run
        sdv = Mock()
        sdv.metadata.get_table_names.return_value = table_names
        sdv.metadata.get_parents.side_effect = parents

        SDV._check_unsupported_dataset_structure(sdv)

        # Asserts
        expect_get_parents_call_count = 3
        assert sdv.metadata.get_parents.call_count == expect_get_parents_call_count

    def test__check_unsupported_dataset_structure_raise_error(self):
        """Test that a ValueError is raised because the bad structure"""
        # Setup
        table_names = ['foo', 'bar', 'tar']
        parents = [[], [], ['foo', 'bar']]

        # Run & assert
        sdv = Mock()
        sdv.metadata.get_table_names.return_value = table_names
        sdv.metadata.get_parents.side_effect = parents

        with pytest.raises(ValueError):
            SDV._check_unsupported_dataset_structure(sdv)

    @patch('sdv.sdv.Sampler')
    @patch('sdv.sdv.Modeler')
    @patch('sdv.sdv.Metadata')
    def test_fit(self, mock_metadata, mock_modeler, mock_sampler):
        """Test fit without the root_path argument"""
        # Run
        metadata = '/path/to/metadata'

        sdv_mock = Mock()
        sdv_mock.model = None
        sdv_mock.distribution = None
        sdv_mock.model_kwargs = None

        SDV.fit(sdv_mock, metadata, root_path='test/path')

        # Asserts
        mock_metadata.assert_called_once_with(
            '/path/to/metadata',
            'test/path'
        )
        mock_modeler.assert_called_once_with(
            metadata=mock_metadata.return_value,
            model=None,
            distribution=None,
            model_kwargs=None
        )
        mock_sampler.assert_called_once_with(
            mock_metadata.return_value,
            mock_modeler.return_value
        )

    def test_sample_rows_fitted(self):
        """Check that the sample_rows is called."""
        # Run
        sdv = Mock()
        table_name = 'DEMO'
        num_rows = 5
        sdv.sampler.sample_rows.return_value = 'test'

        result = SDV.sample_rows(sdv, table_name, num_rows)

        # Asserts
        sdv.sampler.sample_rows.assert_called_once_with(
            'DEMO', 5, sample_children=True, reset_primary_keys=False)

        assert result == 'test'

    def test_sample_rows_not_fitted(self):
        """Check that the sample_rows raise an exception when is not fitted."""
        # Run and asserts
        sdv = Mock()
        sdv.sampler = None
        table_name = 'DEMO'
        num_rows = 5

        with pytest.raises(NotFittedError):
            SDV.sample_rows(sdv, table_name, num_rows)

    def test_sample_table_fitted(self):
        """Check that the sample_table is called"""
        # Run
        sdv = Mock()
        table_name = 'DEMO'
        sdv.sampler.sample_table.return_value = 'test'

        result = SDV.sample_table(sdv, table_name)

        # Asserts
        sdv.sampler.sample_table.assert_called_once_with(
            'DEMO', None, reset_primary_keys=False)

        assert result == 'test'

    def test_sample_table_not_fitted(self):
        """Check that the sample_table raise an exception when is not fitted."""
        # Run and asserts
        sdv = Mock()
        sdv.sampler = None
        table_name = 'DEMO'

        with pytest.raises(NotFittedError):
            SDV.sample_table(sdv, table_name)

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
