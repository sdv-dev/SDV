from unittest import TestCase, skipIf

import pandas as pd

from examples.demo import demo_airbnb, demo_biodegradability, demo_telstra

# Ignored tests
SLOW_TESTS = True


@skipIf(SLOW_TESTS, 'slow tests')
class DTTransformerTest(TestCase):

    def test_airbnb_demo(self):
        """Tests that airbnb demo returns something."""
        synthesized_data = demo_airbnb()
        for table_name in synthesized_data:
            table = synthesized_data[table_name]
            self.assertTrue(isinstance(table, pd.DataFrame))

    def test_telstra_demo(self):
        """Tests that telstra demo returns something."""
        synthesized_data = demo_telstra()
        for table_name in synthesized_data:
            table = synthesized_data[table_name]
            self.assertTrue(isinstance(table, pd.DataFrame))

    def test_bio_demo(self):
        """Tests that bio demo returns something."""
        synthesized_data = demo_biodegradability()
        for table_name in synthesized_data:
            table = synthesized_data[table_name]
            self.assertTrue(isinstance(table, pd.DataFrame))
