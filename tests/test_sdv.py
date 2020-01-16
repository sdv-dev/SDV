from unittest import TestCase
from unittest.mock import Mock, patch

import pytest

from sdv.sdv import DEFAULT_MODEL, DEFAULT_MODEL_KWARGS, SDV, NotFittedError


class TestSDV(TestCase):

    @patch('sdv.sdv.open')
    @patch('sdv.sdv.pickle')
    def test_save(self, pickle_mock, open_mock):
        # Run
        sdv = SDV()
        sdv.save('save/path.pkl')

        # Asserts
        open_mock.assert_called_once_with('save/path.pkl', 'wb')
        output = open_mock.return_value.__enter__.return_value
        pickle_mock.dump.assert_called_once_with(sdv, output)

    @patch('sdv.sdv.open')
    @patch('sdv.sdv.pickle')
    def test_load(self, pickle_mock, open_mock):
        # Run
        returned = SDV.load('save/path.pkl')

        # Asserts
        open_mock.assert_called_once_with('save/path.pkl', 'rb')
        output = open_mock.return_value.__enter__.return_value
        pickle_mock.load.assert_called_once_with(output)
        assert returned is pickle_mock.load.return_value

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

    def test_sample_fitted(self):
        """Check that the sample is called."""
        # Sample
        sdv = Mock()
        sdv.sampler.sample.return_value = 'test'

        # Run
        result = SDV.sample(sdv, 'DEMO', 5)

        # Asserts
        assert result == 'test'
        sdv.sampler.sample.assert_called_once_with(
            'DEMO', 5, sample_children=True, reset_primary_keys=False)

    def test_sample_not_fitted(self):
        """Check that the sample raise an exception when is not fitted."""
        # Setup
        sdv = Mock()
        sdv.sampler = None

        # Run
        with pytest.raises(NotFittedError):
            SDV.sample(sdv, 'DEMO', 5)

    def test_sample_all_fitted(self):
        """Check that the sample_all is called"""
        # Setup
        sdv = Mock()
        sdv.sampler.sample_all.return_value = 'test'

        # Run
        result = SDV.sample_all(sdv)

        # Asserts
        assert result == 'test'
        sdv.sampler.sample_all.assert_called_once_with(5, reset_primary_keys=False)

    def test_sample_all_not_fitted(self):
        """Check that the sample_all raise an exception when is not fitted."""
        # Setup
        sdv = Mock()
        sdv.sampler = None

        # Run
        with pytest.raises(NotFittedError):
            SDV.sample_all(sdv)
