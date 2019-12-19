class SDVModel:
    """Base class for all the models used in ``SDV``.

    The ``SDVModel`` class contains the api methods that must be implemented
    in order to create a new model.
    """

    def fit(self, table_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be fitted.
        """
        raise NotImplementedError

    def sample(self, num_samples):
        """Sample ``num_samples`` rows from the model.

        Args:
            num_samples (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data with the number of rows specified in ``num_samples``.
        """
        raise NotImplementedError

    def get_parameters(self):
        """Get copula model parameters.

        Returns:
            dict:
                Copula flatten parameters.
        """
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Set copula model parameters.

        Args:
            dict:
                Copula flatten parameters.
        """
        raise NotImplementedError
