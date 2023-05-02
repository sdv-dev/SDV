from unittest.mock import Mock, patch

from sdv.metadata.anonymization import (
    _detect_provider_name, get_anonymized_transformer, get_faker_instance, is_faker_function)


class TestAnonimization:

    def test__detect_provider_name(self):
        """Test the ``_detect_provider_name`` method.

        Test that the function returns an expected provider name from the ``faker.Faker`` instance.
        If this is from the ``BaseProvider`` it should also return that name.

        Input:
            - Faker function name.

        Output:
            - The faker provider name for that function.
        """
        # Run / Assert
        email_provider = _detect_provider_name('email')
        lexify_provider = _detect_provider_name('lexify')
        state_provider = _detect_provider_name('state')

        assert email_provider == 'internet'
        assert lexify_provider == 'BaseProvider'
        assert state_provider == 'address.en_US'

    @patch('sdv.metadata.anonymization.AnonymizedFaker')
    def test_get_anonymized_transformer_with_existing_sdtype(self, mock_anonymized_faker):
        """Test the ``get_anonymized_transformer`` method.

        Test that when calling with an existing ``sdtype`` / ``function_name`` from the
        ``SDTYPE_ANONYMZIERS`` dictionary, their ``provider_name`` and ``function_name`` are being
        used by default, and also other ``kwargs`` and provided locales are being passed to the
        ``AnonymizedFaker``.

        Input:
            - ``function_name`` from the ``SDTYPE_ANONYMIZERS``.
            - ``function_kwargs`` additional keyword arguments for that set of arguments.

        Mock:
            - Mock ``AnonymizedFaker`` and assert that has been called with the expected
              arguments.

        Output:
            - The return value must be the instance of ``AnonymizedFaker``.
        """
        # Setup
        output = get_anonymized_transformer('email', transformer_kwargs={
            'function_kwargs': {'domain': '@gmail.com'}, 'locales': ['en_CA', 'fr_CA']
        })
        # Assert
        assert output == mock_anonymized_faker.return_value
        mock_anonymized_faker.assert_called_once_with(
            provider_name='internet',
            function_name='email',
            function_kwargs={'domain': '@gmail.com'},
            locales=['en_CA', 'fr_CA']
        )

    @patch('sdv.metadata.anonymization.AnonymizedFaker')
    def test_get_anonymized_transformer_with_custom_sdtype(self, mock_anonymized_faker):
        """Test the ``get_anonymized_transformer`` method.

        Test that when calling with a custom ``sdtype`` / ``function_name`` that does not belong
        to the ``SDTYPE_ANONYMZIERS`` dictionary. The ``provider_name`` is being found
        automatically other ``kwargs`` and provided locales are being passed to the
        ``AnonymizedFaker``.

        Input:
            - ``function_name`` color.
            - ``function_kwargs`` a dictionary with ``'hue': 'red'``.

        Mock:
            - Mock ``AnonymizedFaker`` and assert that has been called with the expected
              arguments.

        Output:
            - The return value must be the instance of ``AnonymizedFaker``.
        """
        # Setup
        output = get_anonymized_transformer('color', transformer_kwargs={
            'function_kwargs': {'hue': 'red'}, 'locales': ['en_CA', 'fr_CA']
        })

        # Assert
        assert output == mock_anonymized_faker.return_value
        mock_anonymized_faker.assert_called_once_with(
            provider_name='color',
            function_name='color',
            function_kwargs={'hue': 'red'},
            locales=['en_CA', 'fr_CA']
        )

    @patch('sdv.metadata.anonymization.Faker')
    def test_is_faker_function(self, faker_mock):
        """Test that the method returns True if the ``function_name`` is a valid faker function.

        This test mocks the ``Faker`` method to make sure that the ``function_name`` is an
        attribute it has.
        """
        # Setup
        faker_mock.return_value = Mock(spec=['address'])

        # Run
        result = is_faker_function('address')

        # Assert
        assert result is True

    @patch('sdv.metadata.anonymization.get_faker_instance')
    def test_is_faker_function_error(self, mock_get_faker_instance):
        """Test that the method returns False if ``function_name`` is not a valid faker function.

        If the ``function_name`` is not an attribute of ``Faker()`` then we should return false.
        This test mocks ``Faker`` to not have the attribute that is passed as ``function_name``.
        """
        # Setup
        mock_get_faker_instance.return_value = Mock(spec=[])

        # Run
        result = is_faker_function('blah')

        # Assert
        assert result is False
        mock_get_faker_instance.assert_called_once()

    @patch('sdv.metadata.anonymization.Faker')
    def test_get_faker_instance(self, mock_faker):
        """Test that ``get_faker_instance`` returns the same object."""
        # Setup
        first_instance = get_faker_instance()

        # Run
        second_instance = get_faker_instance()

        # Assert
        assert id(first_instance) == id(second_instance)
