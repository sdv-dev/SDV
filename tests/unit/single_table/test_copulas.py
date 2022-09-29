import pytest
from copulas.univariate import BetaUnivariate, GammaUnivariate, UniformUnivariate

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestGaussianCopulaSynthesizer:

    def test__validate_distribution_str(self):
        """Test that when a ``str`` is passed, the class from the ``DISTRIBUTIONS`` is returned."""
        # Setup
        distribution = 'beta'

        # Run
        result = GaussianCopulaSynthesizer._validate_distribution(distribution)

        # Assert
        assert result == BetaUnivariate

    def test__validate_distribution_not_in_distributions(self):
        """Test that ``ValueError`` is raised when the given distribution is not supported."""
        # Setup
        distribution = 'student'

        # Run and Assert
        with pytest.raises(ValueError, match="Invalid distribution specification 'student'."):
            GaussianCopulaSynthesizer._validate_distribution(distribution)

    def test___init__(self):
        """Test creating an instance of ``GaussianCopulaSynthesizer``."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = True
        enforce_rounding = True
        numerical_distributions = None
        default_distribution = None

        # Run
        instance = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance.numerical_distributions == {}
        assert instance.default_distribution == 'beta'
        assert instance._default_distribution == BetaUnivariate
        assert instance._numerical_distributions == {}

    def test___init__custom(self):
        """Test creating an instance of ``GaussianCopulaSynthesizer`` with custom parameters."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = False
        enforce_rounding = False
        numerical_distributions = {'field': 'gamma'}
        default_distribution = 'uniform'

        # Run
        instance = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.numerical_distributions == {'field': 'gamma'}
        assert instance.default_distribution == 'uniform'
        assert instance._default_distribution == UniformUnivariate
        assert instance._numerical_distributions == {'field': GammaUnivariate}

    def test_get_params(self):
        """Test that inherited method ``get_params`` returns all the specific init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = GaussianCopulaSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {},
            'default_distribution': 'beta'
        }
