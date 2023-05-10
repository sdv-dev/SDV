from faker import Faker

from sdv.metadata.anonymization import is_faker_function


def test_is_faker_function():
    """Test is_faker_function checks if function is a valid Faker function."""
    # Run
    result = is_faker_function('address')

    # Assert
    assert result is True


def test_is_faker_function_non_default_locale():
    """Test is_faker_function checks non-default locales."""
    # Setup
    function_name = 'postcode_in_province'

    # Run
    result = is_faker_function(function_name)

    # Assert
    assert result is True
    assert not hasattr(Faker(), function_name)
