import os
import boto3
import botocore
import zipfile

from botocore import UNSIGNED
from botocore.client import Config


BUCKET_NAME = 'hdi-demos'
SDV_NAME = 'sdv-demo'
SUFFIX = '.zip'


def download_folder(folder_name):
    """Downloads and extracts demo folder from S3"""
    s3 = boto3.resource('s3', region_name='us-east-1',
                        config=Config(signature_version=UNSIGNED))
    zip_name = folder_name + SUFFIX
    zip_destination = os.path.join('demo', zip_name)
    key = os.path.join(SDV_NAME, zip_name)
    # make sure directory exists
    if not os.path.exists('demo'):
        os.makedirs('demo')
    # try to download files from s3
    try:
        s3.Bucket(BUCKET_NAME).download_file(key, zip_destination)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    # unzip folder
    zip_ref = zipfile.ZipFile(zip_destination, 'r')
    zip_ref.extractall('demo')
    zip_ref.close()
