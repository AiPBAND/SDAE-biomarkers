import functools

import numpy as np
import tensorflow as tf

from google.cloud import storage


def download_blob(bucket_name, source_blob_name, file_path):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(file_path)

    print("Blob {} downloaded to {}.".format(source_blob_name, file_path))


def load_gs_data(bucket_name, source_blob_name, file_path="./data/data.csv", 
    batch_size=5, num_epochs=10, shuffle=True, **kwargs):
    
    print("Initializing TF dataset...")
    
    download_blob(bucket_name, source_blob_name, file_path)

    dataset = tf.data.experimental.make_csv_dataset(file_path, 
        batch_size=batch_size, num_epochs=num_epochs, **kwargs)
    
    reconstruction_dataset = tf.data.Dataset.zip((dataset,dataset))
    
    return dataset, reconstruction_dataset
    
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))