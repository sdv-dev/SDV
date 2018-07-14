import os
import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config


BUCKET_NAME = 'hdi-demos'  # replace with your bucket name
sessions_KEY = 'sdv-demo/sessions_demo.csv'
users_KEY = 'sdv-demo/users_demo.csv'
meta_KEY = 'sdv-demo/Airbnb_demo_meta.json'

s3 = boto3.resource('s3', region_name='us-east-1',
                    config=Config(signature_version=UNSIGNED))

# make sure directory exists
if not os.path.exists('demo'):
    os.makedirs('demo')

# try to download files from s3
try:
    s3.Bucket(BUCKET_NAME).download_file(sessions_KEY,
                                         'demo/sessions_demo.csv')
    s3.Bucket(BUCKET_NAME).download_file(users_KEY, 'demo/users_demo.csv')
    s3.Bucket(BUCKET_NAME).download_file(meta_KEY,
                                         'demo/Airbnb_demo_meta.json')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
