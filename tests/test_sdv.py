from unittest import TestCase
from unittest.mock import Mock, patch

import pytest

from sdv.errors import NotFittedError
from sdv.sdv import SDV


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
        assert sdv._model == sdv.DEFAULT_MODEL
        assert sdv._model_kwargs == sdv.DEFAULT_MODEL_KWARGS
        assert sdv._model_kwargs is not sdv.DEFAULT_MODEL_KWARGS

    def test____init__users_params(self):
        """Create default instance"""
        # Run
        sdv = SDV(model='test', model_kwargs={'a': 2})

        # Asserts
        assert sdv._model == 'test'
        assert sdv._model_kwargs == {'a': 2}

    def test_sample_fitted(self):
        """Check that the sample is called."""
        # Sample
        sdv = Mock()
        sdv._model_instance.sample.return_value = 'test'

        # Run
        result = SDV.sample(sdv, 'DEMO', 5)

        # Asserts
        assert result == 'test'
        sdv._model_instance.sample.assert_called_once_with(
            'DEMO', 5, sample_children=True, reset_primary_keys=False)

    def test_sample_not_fitted(self):
        """Check that the sample raise an exception when is not fitted."""
        # Setup
        sdv = SDV()

        # Run
        with pytest.raises(NotFittedError):
            sdv.sample('DEMO', 5)
