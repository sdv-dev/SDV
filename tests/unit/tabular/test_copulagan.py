from unittest.mock import Mock, call, patch

import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import GaussianCopulaTransformer

from sdv.tabular.copulagan import CopulaGAN
from tests.utils import DataFrameMatcher


class TestCopulaGAN:

    @patch('sdv.tabular.copulagan.CTGAN._fit')
    @patch('sdv.tabular.copulagan.HyperTransformer', spec_set=HyperTransformer)
    @patch('sdv.tabular.copulagan.GaussianCopulaTransformer',
           spec_set=GaussianCopulaTransformer)
    def test__fit(self, gct_mock, ht_mock, ctgan_fit_mock):
        """Test the ``CopulaGAN._fit`` method.

        The ``_fit`` method is expected to:
        - Build transformers for all the non-categorical data columns based on the
          field distributions.
        - Create a HyperTransformer with all the transformers.
        - Fit and transform the data with the HyperTransformer.
        - Call CTGAN fit.

        Setup:
            - mock _field_distribution and _default_distribution to return the desired
              distribution values

        Input:
            - pandas.DataFrame

        Expected Output:
            - None

        Side Effects:
            - GaussianCopulaTransformer is called with the expected disributions.
            - HyperTransformer is called to create a hyper transformer object.
            - HyperTransformer fit_transform is called with the expected data.
            - CTGAN's fit method is called with the expected data.
        """
        # Setup
        model = Mock(spec_set=CopulaGAN)
        model._field_distributions = {'a': 'a_distribution'}
        model._default_distribution = 'default_distribution'
        model._metadata.get_fields.return_value = {'a': {}, 'b': {}, 'c': {'type': 'categorical'}}

        # Run
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [5, 6, 7],
            'c': ['c', 'c', 'c'],
        })
        out = CopulaGAN._fit(model, data)

        # asserts
        assert out is None
        assert model._field_distributions == {'a': 'a_distribution'}
        gct_mock.assert_has_calls([
            call(distribution='a_distribution'),
            call(distribution='default_distribution'),
        ])
        assert gct_mock.call_count == 2

        assert model._ht == ht_mock.return_value
        ht_mock.return_value.fit_transform.called_once_with(DataFrameMatcher(data))
        ctgan_fit_mock.called_once_with(DataFrameMatcher(data))

    @patch('sdv.tabular.copulagan.CTGAN._fit')
    @patch('sdv.tabular.copulagan.HyperTransformer', spec_set=HyperTransformer)
    @patch('sdv.tabular.copulagan.GaussianCopulaTransformer',
           spec_set=GaussianCopulaTransformer)
    def test__fit_with_transformed_columns(self, gct_mock, ht_mock, ctgan_fit_mock):
        """Test the ``CopulaGAN._fit`` method with transformed columns.

        The ``_fit`` method is expected to:
        - Build transformers for all the data columns based on the field distributions.
        - Create a HyperTransformer with all the transformers.
        - Fit and transform the data with the HyperTransformer.
        - Call CTGAN fit.

        Setup:
            - mock _field_distribution and _default_distribution to return the desired
              distribution values

        Input:
            - pandas.DataFrame

        Expected Output:
            - None

        Side Effects:
            - GaussianCopulaTransformer is called with the expected disributions.
            - HyperTransformer is called to create a hyper transformer object.
            - HyperTransformer fit_transform is called with the expected data.
            - CTGAN's fit method is called with the expected data.
        """
        # Setup
        model = Mock(spec_set=CopulaGAN)
        model._field_distributions = {'a': 'a_distribution'}
        model._default_distribution = 'default_distribution'
        model._metadata.get_fields.return_value = {'a': {}}

        # Run
        data = pd.DataFrame({
            'a.value': [1, 2, 3]
        })
        out = CopulaGAN._fit(model, data)

        # asserts
        assert out is None
        assert model._field_distributions == {'a': 'a_distribution'}
        gct_mock.assert_called_once_with(distribution='a_distribution')

        assert model._ht == ht_mock.return_value
        ht_mock.return_value.fit_transform.called_once_with(DataFrameMatcher(data))
        ctgan_fit_mock.called_once_with(DataFrameMatcher(data))
