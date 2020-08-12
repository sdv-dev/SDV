from unittest import TestCase

import pandas as pd

# from unittest.mock import Mock, call
# from sdv.constraints.base import Constraint, _get_qualified_name


class ConstraintsTest(TestCase):
    # from unittest.mock import Mock, call
    # from sdv.constraints.base import Constraint, _get_qualified_name

    def SetUp(self):
        self.table_data = pd.DataFrame({})

    def _test_get_qualified_name_attr(self):
        """Checks if the returned Fully Qualified Name is the expected,
        if object has the given named attribute."""

    def _test_get_qualified_name_no_attr(self):
        """Checks if the returned Fully Qualified Name is the expected,
        if object has not the given named attribute."""

    def test_fit(self):
        """It checks if method `fit` is correctly called."""

    def test_transform(self):
        """It calls `transform` and checks if the return is identycal
        to the table_data passed."""

    def test_fit_transform(self):
        """With Mock, it makes sure methods `fit` and `transform`
        are correctly called."""

    def test_reverse_transform(self):
        """It calls `reverse_transform` and checks if the return is identycal
        to the table_data passed."""

    def test_is_valid(self):
        """Given a table data, it checks if the resulting series of
        "True" values is corect."""

    def test_filter_valid(self):
        """Given a table data, it checks if the resulting
        valid rows are corect. Mith Mock, it makes sure method `is valid` is
        called correctly."""

    def test_filter_valid_zero_rows(self):
        """Given a table data and a constraint, it checks if the resulting
        valid rows are corect if no column satisfy the constraint.
        Mith Mock, it makes sure method `is valid` is called correctly."""

    def test_to_dict(self):
        """Checks if the returning dict is correct."""
