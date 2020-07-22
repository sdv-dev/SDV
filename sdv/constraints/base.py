class Constraint:
    """Constraint base class."""

    def fit(self, data):
        """No-op method."""
        pass

    def transform(self, data):
        """Identity method for completion. To be optionally overwritten by subclasses.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        return data

    def fit_transform(self, data):
        """Fit this Constraint to the data and then transform it.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Identity method for completion. To be optionally overwritten by subclasses.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        return data

    def filter_valid(self, data):
        """Identity method for completion. To be optionally overwritten by subclasses.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Input data unmodified.
        """
        return data


class TransformationConstraint(Constraint):

    def __init__(self, transform=None, reverse_transform=None, copy=True):
        self._transform = transform
        self._reverse_transform = reverse_transform
        self._copy = copy

    def transform(self, data):
        if self._copy:
            data = data.copy()

        return self._transform(data)

    def reverse_transform(self, data):
        if self._copy:
            data = data.copy()

        return self._reverse_transform(data)


class ValidationConstraint(Constraint):

    def __init__(self, validation):
        self._validation = validation

    def filter_valid(self, data):
        return data[self._validation(data)]
