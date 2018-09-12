import logging
import os
import sys

from sdv.sdv import SDV
from timeit import default_timer as timer
from utils import download_folder


def get_logger():
    logger = logging.getLogger()
    # We create a formatter
    fmt = '%(asctime)s - %(process)d - %(module)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)

    # Setup Handler
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = get_logger()


table_dict = {
    'airbnb_demo': ['users', 'sessions'],
    'biodegradability': ['molecule', 'atom', 'bond'],
    'coupon_purchase_prediction': ['prefecture_locations', 'user_list',
                                   'coupon_list_train', 'coupon_visit_train',
                                   'coupon_detail_train', 'coupon_area_train'],
    'hilary_clinton_emails': ['Persons', 'Emails', 'Aliases',
                              'EmailReceivers'],
    'mutagenesis': ['molecule', 'atom', 'bond'],
    'nips_2015_papers': ['Authors', 'Papers', 'PaperAuthors'],
    'rossman': ['store', 'train'],
    'telstra': ['severity_type', 'resource_type',
                'event_type', 'log_feature'],
    'walmart': ['stores', 'train', 'features'],
    'world_development_indicators': ['Country', 'Series',
                                     'CountryNotes', 'Footnotes',
                                     'SeriesNotes', 'Indicators']
}


def run_demo(folder_name):
    """Runs the demo for specified folder"""
    start = timer()
    meta_file = os.path.join(
        'demo', folder_name, folder_name.capitalize() + '_manual_meta.json')
    sdv = SDV(meta_file)
    sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                sdv.dn.transformed_data)
    table_list = table_dict[folder_name]
    for table in table_list:
        sampled_rows[table] = sdv.sample_rows(table, 1)
        LOGGER.info('Sampled row from %s: %s', table, sampled_rows[table])
    end = timer()
    LOGGER.info('Total time: %s seconds', round(end-start))


if __name__ == '__main__':
    download_folder(sys.argv[1])
    run_demo(sys.argv[1])
