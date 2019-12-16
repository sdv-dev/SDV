class SDVModel:

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
        raise NotImplementedError

    def set_parameters(self, parameters):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, copula_dict):
        raise NotImplementedError
