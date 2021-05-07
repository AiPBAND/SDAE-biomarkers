# Import comet_ml at the top of your file
import tensorflow as tf 
import argparse
import pandas as pd
import numpy as np
from models import Autoencoder
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import os 
import wandb
from wandb.keras import WandbCallback
import random
from tempfile import TemporaryFile
import gzip
from collections import Counter
from pprint import pprint
#os.environ["WANDB_MODE"] = "offline"

SEED = 0


parser = argparse.ArgumentParser() 
parser.add_argument(
    "--data-path",
    type=str,
    dest="FILE_PATH",
    help="Path to the CSV input data. First row headers. First column IDs",
    default="data\compressed\gene_exp_MA.csv.gz",
)

parser.add_argument(
    "--test",
    default=0.2,
    dest="TEST_RATIO",
    type=float,
    help="Ratio of samples kept out for testing.",
)

parser.add_argument(
    "--validate",
    default=0.1,
    dest="VALIDATION_RATIO",
    type=float,
    help="Number of samples kept out for validation.",
)

parser.add_argument(
    "--verbose",
    default=1,
    dest="VERBOSITY",
    type=int,
    choices=[0, 1, 2],
    help="Verbosity level: 0 None, 1 Info, 2 All",
)

wandb.login() 
run = wandb.init(project='SDAE-biomarkers', 
           entity='aipband', 
           job_type="data_prep",
           notes="Train, validation, test splits.", 
           tags=["baseline", "cross-validation", "data"]) 

args=parser.parse_args()
wandb.config.update(args)
wandb.log({'random_seed': SEED})

with gzip.open(args.FILE_PATH, 'rb') as f_in: 
    data = pd.read_csv(f_in, index_col=0) 
dataframe = data.values[:,1:-2]
labels = data.values[:,-1]    
dataframe = normalize(data)

pprint({"total_samples": len(dataframe)})

sss = StratifiedShuffleSplit(n_splits=1, test_size=args.VALIDATION_RATIO, random_state=SEED)
i_train_test, i_validation = next(sss.split(dataframe, labels))

train_test_set, validation = dataframe[i_train_test], dataframe[i_validation]
train_test_labels, validation_labels = labels[i_train_test], labels[i_validation]
pprint({'validation_indices': i_validation, 
           'validation_labels':Counter(validation_labels), 
           'validation_ratio': args.VALIDATION_RATIO, 
           'validation_count': len(validation_labels)})

sss = StratifiedShuffleSplit(n_splits=5, test_size=args.TEST_RATIO, random_state=SEED)
sss = sss.split(dataframe, labels)

data = {"train":[], 
        "test":[], 
        "train_labels":[], 
        "test_labels":[],
        "validation": validation,
        "validation_labels":validation_labels}
for i, (train_i, test_i) in enumerate(sss):
    
    pprint({'split': {
        'split_id': i,
        'data_labels':Counter(labels[test_i]), 
        'test_count': len(test_i)}})
    
    data["train"].append(dataframe[train_i])
    data["test"].append(dataframe[test_i])
    data["train_labels"].append(labels[train_i])
    data["test_labels"].append(labels[test_i])

artifact = wandb.Artifact("data_splits", type="dataset")

np.savez("temp_file", data) 
artifact.add_file("temp_file.npz", name="split_data") 
wandb.run.log_artifact(artifact) 

