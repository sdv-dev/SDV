import io
import logging
import os
import urllib
import zipfile

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
