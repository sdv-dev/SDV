class Modeler:
    """ Class responsible for modeling database """
    def __init__(self, data_navigator, transformed_data,
                 model_type='Gaussian'):
        """ Instantiates a modeler object
        Args:
            data_navigator: A DataNavigator object for the dataset
            transformed_data: tables post tranformation {table_name:dataframe}
            model_type: Type of Copula to use for modeling
        """
        pass

    def CPA(self, table):
        """ Runs CPA algorithm on a table
        Args:
            table (string): name of table
        """
        pass

    def RCPA(self, table):
        """ Recursively calls CPA starting at table
        Args:
            table (string): name of table to start from
        """
        pass

    def model_database(self):
        """ Uses RCPA and stores model for database """
        pass

    def load_model(self, filename):
        """ Loads model from filename
        Args:
            filename (string): path of file to load
        """
        pass

    def save_model(self, file_destination):
        """ Saves model to file destination
        Args:
            file_destination (string): path to store file
        """
        pass
