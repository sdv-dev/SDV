from unittest.mock import Mock

import numpy as np
import pandas as pd

from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import get_multi_table_data, get_multi_table_metadata


class TestHMASynthesizer:

    def test___init__(self):
        """Test the default initialization of the ``HMASynthesizer``."""
        # Run
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()
        instance = HMASynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == {
            'nesreca': {'default_distribution': 'beta'},
            'oseba': {'default_distribution': 'beta'},
            'upravna_enota': {'default_distribution': 'beta'},
        }
        instance.metadata.validate.assert_called_once_with()

    def test__get_extension(self):
        """Test the ``_get_extension`` method.

        Test that the resulting dataframe contains extended columns using the names
        and parameters from a trained ``copulas.univariate`` model.
        """
        # Setup
        metadata = get_multi_table_metadata()
        child_table = pd.DataFrame({
            'id_nesreca': [0, 1, 2, 3],
            'upravna_enota': [0, 1, 2, 3]
        })
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_extension('nesreca', child_table, 'upravna_enota')

        # Assert
        expected = pd.DataFrame({
            '__nesreca__upravna_enota__univariates__id_nesreca__a': [1., 1., 1., 1.],
            '__nesreca__upravna_enota__univariates__id_nesreca__b': [1., 1., 1., 1.],
            '__nesreca__upravna_enota__univariates__id_nesreca__loc': [0., 1., 2., 3.],
            '__nesreca__upravna_enota__univariates__id_nesreca__scale': [np.nan] * 4,
            '__nesreca__upravna_enota__num_rows': [1., 1., 1., 1.]
        })

        pd.testing.assert_frame_equal(result, expected)

    def test__get_foreign_keys(self):
        """Test that this method returns the foreign keys for a given table name and child name."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_foreign_keys('nesreca', 'oseba')

        # Assert
        assert result == ['id_nesreca']

    def test__get_all_foreign_keys(self):
        """Test that this method returns all the foreign keys for a given table name."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)

        # Run
        result = instance._get_all_foreign_keys('nesreca')

        # Assert
        assert result == ['upravna_enota']

    def test__extend_table(self):
        """Test that ``extend_table`` extends the current table with extra columns.

        This also updates ``self._modeled_tables`` and ``self._max_child_rows``.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)
        metadata.add_column('nesreca', 'value', sdtype='numerical')
        metadata.add_column('oseba', 'oseba_value', sdtype='numerical')
        metadata.add_column('upravna_enota', 'name', sdtype='categorical')

        data = get_multi_table_data()
        data['nesreca']['value'] = [0, 1, 2, 3]
        data['oseba']['oseba_value'] = [0, 1, 2, 3]

        # Run
        result = instance._extend_table(data['nesreca'], data, 'nesreca')

        # Assert
        expected_result = pd.DataFrame({
            'id_nesreca': [0, 1, 2, 3],
            'upravna_enota': [0, 1, 2, 3],
            'value': [0, 1, 2, 3],
            '__oseba__id_nesreca__covariance__0__0': [0.] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__a': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__b': [1.] * 4,
            '__oseba__id_nesreca__univariates__oseba_value__loc': [0., 1., 2., 3.],
            '__oseba__id_nesreca__univariates__oseba_value__scale': [np.nan] * 4,
            '__oseba__id_nesreca__univariates__upravna_enota__a': [1.] * 4,
            '__oseba__id_nesreca__univariates__upravna_enota__b': [1.] * 4,
            '__oseba__id_nesreca__univariates__upravna_enota__loc': [0., 1., 2., 3.],
            '__oseba__id_nesreca__univariates__upravna_enota__scale': [np.nan] * 4,
            '__oseba__id_nesreca__num_rows': [1.] * 4,
        })

        pd.testing.assert_frame_equal(expected_result, result)
        assert instance._modeled_tables == ['oseba']
        assert instance._max_child_rows['__oseba__id_nesreca__num_rows'] == 1

    def test__pop_foreign_keys(self):
        """Test that this method removes the foreign keys from the ``table_data``."""
        # Setup
        instance = Mock()
        instance._get_all_foreign_keys.return_value = ['a', 'b']
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [2, 3, 4],
            'c': ['John', 'Doe', 'Johanna']
        })

        # Run
        result = HMASynthesizer._pop_foreign_keys(instance, table_data, 'table_name')

        # Assert
        pd.testing.assert_frame_equal(pd.DataFrame({'c': ['John', 'Doe', 'Johanna']}), table_data)
        np.testing.assert_array_equal(result['a'], [1, 2, 3])
        np.testing.assert_array_equal(result['b'], [2, 3, 4])

    def test__clear_nans(self):
        """Test that this method clears all the nans and substitutes them with expected values."""
        # Setup
        data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, np.nan, np.nan],
            'categorical': ['John', np.nan, 'Johanna', 'John', np.nan, 'Doe'],
        })

        # Run
        HMASynthesizer._clear_nans(data)

        # Assert
        expected_data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, 1.5, 1.5],
            'categorical': ['John', 'John', 'Johanna', 'John', 'John', 'Doe']
        })
        pd.testing.assert_frame_equal(expected_data, data)

    def test__model_table(self):
        """Test that ``_model_table`` performs the modeling.

        Modeling consists of getting the table for the given table name,
        learning the size of this table, removing the foreign keys and clearing
        any null values by using the ``_clear_nans`` method. Then, fitting the table model by
        calling ``fit_processed_data``,  adding back the foreign keys, updating the ``tables`` and
        marking the table name as modeled within the ``instance._modeled_tables``.
        """
        # Setup
        nesreca_model = Mock()
        instance = Mock()
        instance._synthesizer = GaussianCopulaSynthesizer

        instance._modeled_tables = []
        instance._table_sizes = {}
        instance._table_synthesizers = {'nesreca': nesreca_model}
        instance._pop_foreign_keys.return_value = {'fk': [1, 2, 3]}
        data = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2]
        })
        extended_data = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2],
            'extended': ['a', 'b', 'c']
        })

        instance._extend_table.return_value = extended_data

        tables = {'nesreca': data}

        # Run
        result = HMASynthesizer._model_table(instance, 'nesreca', tables)

        # Assert
        expected_result = pd.DataFrame({
            'id_nesreca': [0, 1, 2],
            'upravna_enota': [0, 1, 2],
            'extended': ['a', 'b', 'c'],
            'fk': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(expected_result, result)

        instance._extend_table.assert_called_once_with(data, tables, 'nesreca')
        instance._pop_foreign_keys.assert_called_once_with(extended_data, 'nesreca')
        instance._clear_nans(extended_data)
        nesreca_model.fit_processed_data.assert_called_once_with(extended_data)

        assert instance._modeled_tables == ['nesreca']
        assert instance._table_sizes == {'nesreca': 3}

    def test__fit(self):
        """Test that ``_fit`` calls ``_model_table`` only if the table has no parents."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = HMASynthesizer(metadata)
        instance._model_table = Mock()
        data = get_multi_table_data()
        data['nesreca']['value'] = [0, 1, 2, 3]
        data['oseba']['oseba_value'] = [0, 1, 2, 3]

        # Run
        instance._fit(data)

        # Assert
        instance._model_table.assert_called_once_with('upravna_enota', data)
