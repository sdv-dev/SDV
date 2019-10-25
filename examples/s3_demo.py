import io
import logging
import os
import sys
import urllib
import zipfile
from timeit import default_timer as timer

from sdv.sdv import SDV

LOGGER = logging.getLogger(__name__)

BUCKET_NAME = 'hdi-demos'
SDV_NAME = 'sdv-demo'
SUFFIX = '.zip'


def download_folder(folder_name):
    """Downloads and extracts demo folder from S3"""
    zip_name = folder_name + SUFFIX
    key = os.path.join(SDV_NAME, zip_name)

    # If the directory doesn't exist , we create it
    # If it exists, we check for the folder_name for early exit
    if not os.path.exists('demo'):
        os.makedirs('demo')

    else:
        if os.path.exists(os.path.join('demo', folder_name)):
            return

    # try to download files from s3
    try:
        url = 'https://{}.s3.amazonaws.com/{}'.format(BUCKET_NAME, key)
        response = urllib.request.urlopen(url)
        bytes_io = io.BytesIO(response.read())

    except urllib.error.HTTPError as error:
        if error.code == 404:
            LOGGER.error('File %s not found.', key)
        raise

    # unzip folder
    zip_ref = zipfile.ZipFile(bytes_io, 'r')
    zip_ref.extractall('demo')
    zip_ref.close()


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
    sampled = sdv.sample_all()

    LOGGER.info('Parent map: %s', sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s', sdv.dn.transformed_data)

    for name, table in sampled.items():
        LOGGER.info('Sampled row from %s: %s', name, table.head(3).T)

    end = timer()
    LOGGER.info('Total time: %s seconds', round(end - start))


if __name__ == '__main__':
    fmt = '%(asctime)s - %(process)d - %(module)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    download_folder(sys.argv[1])
    run_demo(sys.argv[1])
