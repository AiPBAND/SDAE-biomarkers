import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
from google.cloud import storage
import os
import logging

logger = logging.getLogger("SDAE")


def download_blob(bucket_name, source_blob_name, file_path):
    logger.info("Downloading data files from GS storage.")

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(file_path)

    logger.info("File {} downloaded to {}.".format(source_blob_name, file_path))


def load_gs_data(bucket_name, source_blob_name, file_path, **kwargs):
    
    logger.info("Initializing dataset...")
    
    save_path = os.path.join(file_path, source_blob_name)
    
    download_blob(bucket_name, source_blob_name, save_path)

    dataframe = pd.read_csv(save_path, header=0, index_col=0)
    logger.info("Loaded {} samples with {} features.".format(dataframe.shape[0], dataframe.shape[1]))

    return dataframe
    