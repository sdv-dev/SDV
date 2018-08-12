import unittest
import demo_data_downloader

from demo import demo_airbnb, demo_telstra, demo_biodegradability


class DTTransformerTest(unittest.TestCase):

    def test_airbnb_demo(self):
        demo_airbnb()

    def test_telstra_demo(self):
        demo_telstra()

    def test_bio_demo(self):
        demo_biodegradability()


if __name__ == '__main__':
    unittest.main()
