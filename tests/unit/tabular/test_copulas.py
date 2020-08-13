from unittest.mock import Mock, patch
import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate
from sdv.tabular.copulas import GaussianCopula


class TestGaussianCopula:

    @patch('sdv.tabular.copulas.copulas.multivariate.GaussianMultivariate',
           spec_set=GaussianMultivariate)
    def test__fit(self, gm_mock):
        # setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._distribution = 'a_distribution'

        # run
        data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        out = GaussianCopula._fit(gaussian_copula, data)

        # asserts
        assert out is None
        gm_mock.assert_called_once_with(distribution='a_distribution')

        assert gaussian_copula._model == gm_mock.return_value
        expected_data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        call_args = gaussian_copula._model.fit.call_args_list
        passed_table_data = call_args[0][0][0]

        pd.testing.assert_frame_equal(expected_data, passed_table_data)

        gaussian_copula._update_metadata.assert_called_once_with()

    def test__rebuild_covariance_matrix_positive_definite():
        # Run
        covariance = [[1],[0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[1., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_covariance_matrix_not_positive_definite():
        # Run
        covariance = [[-1],[0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[0., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_gaussian_copula():
        # Setup
        sdvmodel = Mock(autospec=GaussianCopula)
        sdvmodel._rebuild_covariance_matrix.return_value = [[0.4, 0.17], [0.17, 0.07]]
        sdvmodel._distribution= {'foo':'GaussianUnivariate'}

        # Run
        model_parameters = {
        'univariates': {
            'foo': {
                'scale': 0.0,
                'loc': 5
            },
        },
        'covariance': [[0.1],[0.4, 0.1]],
        'distribution': 'GaussianUnivariate',

        }
        result = GaussianCopula._rebuild_gaussian_copula(sdvmodel, model_parameters)

        # Asserts
        expected = {
        'univariates': [
            {
                'scale': 1.0,
                'loc': 5,
                'type': 'GaussianUnivariate'
            }
        ],
        'columns': ['foo'],
        'distribution': 'GaussianUnivariate',
        'covariance': [[0.4, 0.17], [0.17, 0.07]]
        }
        assert result == expected

    def test__set_parameters_negative_max_rows():
        pass

    def test_get_parameters():
        pass

    def test_get_parameters_non_parametric():
        pass

    def test__sample_positive():
        pass

    def test__sample_negative():
        pass

    def test__sample_float():
        pass

    def test__update_metadata():
        pass

    def test_get_distribution_not_distribution():
        pass

    def test_get_distribution_str_distribution():
        pass

    def test_get_distribution_dict_distribution():
        pass

    """
    WORK IN PROGRESS
    def test__set_parameters():
        #WORK IN PROGRESS, IS NECESARY COMPARE THE VALUES ONE BY ONE
        # Setup
        flat = {
            'foo__0__foo': 'foo value',
            'bar__0__0': 'bar value',
            'tar': 'tar value',
            'num_rows': 1,
            'columns': 1,
            'univariates': '',
            'covariance': ''
        }
        returned = {
            'foo': {0: {'foo': 'foo value'}},
            'bar': [['bar value']],
            'tar': 'tar value',
            'num_rows': 1,
            'columns': 1,
            'univariates': '',
            'covariance': ''
        }
        sdvmodel = Mock(autospec=GaussianCopula)
        sdvmodel._rebuild_gaussian_copula.return_value = returned
        #Run
        GaussianCopula.set_parameters(sdvmodel,flat)
        #Asserts
        result = sdvmodel._model
        init_expected = {'foo': {0: {'foo': 'foo value'}},
             'bar': [['bar value']],
             'tar': 'tar value',
             'columns': 1,
             'univariates': '',
             'covariance': ''
        }
        expected = GaussianMultivariate().from_dict(init_expected)
        expected._num_rows = 1
        #NEEDS FUNCTION TO COMPARE CHANGES
    """
