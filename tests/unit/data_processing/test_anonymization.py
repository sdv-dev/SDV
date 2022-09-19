from unittest.mock import patch

from sdv.data_processing.anonymization import _detect_provider_name, get_anonymized_transformer


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

        assert email_provider == 'internet'
        assert lexify_provider == 'BaseProvider'

    @patch('sdv.data_processing.anonymization.AnonymizedFaker')
    def test_get_anonymized_transformer_with_existing_sdtype(self, mock_anonymized_faker):
        """Test the ``get_anonymized_transformer`` method.

        Test that when calling with an existing ``sdtype`` / ``function_name`` from the
        ``SDTYPE_ANONYMZIERS`` dictionary, their ``provider_name`` and ``function_name`` are being
        used by default, and also other ``kwargs`` are being passed to the ``AnonymizedFaker``.

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
        output = get_anonymized_transformer('email', function_kwargs={'domain': '@gmail.com'})

        # Assert
        assert output == mock_anonymized_faker.return_value
        mock_anonymized_faker.assert_called_once_with(
            provider_name='internet',
            function_name='email',
            domain='@gmail.com'
        )

    @patch('sdv.data_processing.anonymization.AnonymizedFaker')
    def test_get_anonymized_transformer_with_custom_sdtype(self, mock_anonymized_faker):
        """Test the ``get_anonymized_transformer`` method.

        Test that when calling with a custom ``sdtype`` / ``function_name`` that does not belong
        to the ``SDTYPE_ANONYMZIERS`` dictionary. The ``provider_name`` is being found
        automatically other ``kwargs`` are being passed to the ``AnonymizedFaker``.

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
        output = get_anonymized_transformer('color', function_kwargs={'hue': 'red'})

        # Assert
        assert output == mock_anonymized_faker.return_value
        mock_anonymized_faker.assert_called_once_with(
            provider_name='color',
            function_name='color',
            hue='red'
        )
