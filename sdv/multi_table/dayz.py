"""Multi-Table DayZ parameter detection and creation."""

from sdv.errors import SynthesizerInputError
from sdv.multi_table._dayz_utils import create_parameters_multi_table


class DayZSynthesizer:
    """Multi-Table DayZSynthesizer for public SDV."""

    def __init__(self, metadata, locales=['en_US']):
        raise SynthesizerInputError(
            "Only the 'DayZSynthesizer.create_parameters' is a SDV public feature. "
            'To define and use and use a DayZSynthesizer object you must have an SDV-Enterprise'
            ' version.'
        )

    @classmethod
    def create_parameters(cls, data, metadata, output_filename=None):
        """Create parameters for the DayZSynthesizer.

        Args:
            data (dict[str, pd.DataFrame]): The input data.
            metadata (Metadata): The metadata object.
            output_filename (str, optional): The output filename for the parameters JSON.

        Returns:
            dict: The created parameters.
        """
        return create_parameters_multi_table(data, metadata, output_filename)
