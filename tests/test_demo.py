from unittest import TestCase, skipIf
import demo_data_downloader

from demo import demo_airbnb, demo_telstra, demo_biodegradability


# Ignored tests
SLOW_TESTS = True


@skipIf(SLOW_TESTS, 'slow tests')
class DTTransformerTest(TestCase):

    def test_airbnb_demo(self):
        """Tests that airbnb demo returns something."""
        demo_airbnb()

    def test_telstra_demo(self):
        """Tests that telstra demo returns something."""
        demo_telstra()

    def test_bio_demo(self):
        """Tests that bio demo returns something."""
        demo_biodegradability()
