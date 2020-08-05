import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
from google.cloud import storage
import os


def download_blob(bucket_name, source_blob_name, file_path):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(file_path)

    print("Blob {} downloaded to {}.".format(source_blob_name, file_path))


def load_gs_data(bucket_name, source_blob_name, file_path="./data", **kwargs):
    
    print("Initializing dataset...")
    
    save_path = os.path.join(file_path, source_blob_name)
    
    download_blob(bucket_name, source_blob_name, save_path)

    dataframe = pd.read_csv(save_path, header=0, index_col=0)
    print("Loaded {} samples with {} features.".format(dataframe.shape[0], dataframe.shape[1]))

    return dataframe
    