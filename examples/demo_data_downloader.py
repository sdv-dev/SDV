import os
import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config


BUCKET_NAME = 'hdi-demos'  # replace with your bucket name
bio_KEY = 'sdv-demo/biodegradability.zip'
airbnb_demo_KEY = 'sdv-demo/airbnb_demo.zip'
airbnb_complete_KEY = 'sdv-demo/airbnb_complete.zip'
telstra_KEY = 'sdv-demo/telstra.zip'

s3 = boto3.resource('s3', region_name='us-east-1',
                    config=Config(signature_version=UNSIGNED))

# make sure directory exists
if not os.path.exists('demo'):
    os.makedirs('demo')
    # try to download files from s3
    try:
        s3.Bucket(BUCKET_NAME).download_file(bio_KEY,
                                             'demo/biodegradability.zip')
        s3.Bucket(BUCKET_NAME).download_file(airbnb_demo_KEY,
                                             'demo/airbnb_demo.zip')
        s3.Bucket(BUCKET_NAME).download_file(airbnb_complete_KEY,
                                             'demo/airbnb_complete.zip')
        s3.Bucket(BUCKET_NAME).download_file(telstra_KEY,
                                             'demo/telstra.zip')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
