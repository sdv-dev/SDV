from sdv.constraints.base import Constraint


class ValidationConstraint(Constraint):
    """Base Validation Constraint class.

    This class can be subclassed in order to develop other
    generic validation constraints or used directly by passing
    the validation function as an argument.

    The validation function must have the signature::

        def validation(data: pandas.DataFrame) -> pandas.DataFrame:

    Args:
        validation (callable):
            Function to replace the ``filter_valid`` method.
    """

    def __init__(self, validation):
        self.filter_valid = validation
