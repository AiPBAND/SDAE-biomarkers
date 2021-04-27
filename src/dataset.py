# Import comet_ml at the top of your file
import tensorflow as tf 
import argparse
import pandas as pd
import numpy as np
from models import Autoencoder
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import os
from decouple import config as env_config
import wandb
from wandb.keras import WandbCallback
from random import random
from tempfile import TemporaryFile  
 

os.environ["WANDB_MODE"] = "dryrun"
parser = argparse.ArgumentParser() 
parser.add_argument(
    "--data-path",
    type=str,
    dest="data_path",
    help="Path to the CSV input data. First row headers. First column IDs",
    default="uni_exp_data_MA_7.csv",
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
    dest="VALIDATION_COUNT",
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

artifact = run.use_artifact('input_tables:latest')
artifact_dir = artifact.download()

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

data = pd.read_csv("gene_exp_MA.csv", index_col=0) 
dataframe = data.values[:,1:-2]
labels = data.values[:,-1]    
dataframe = normalize(data)

sss = StratifiedShuffleSplit(n_splits=1, test_size=args.VALIDATION_COUNT, random_state=0)
i_train_test, i_validation = next(sss.split(dataframe, labels))

train_test_set, validation = dataframe[i_train_test], dataframe[i_validation]
train_test_labels, validation_labels = labels[i_train_test], labels[i_validation]

sss = StratifiedShuffleSplit(n_splits=5, test_size=args.TEST_RATIO, random_state=0)
(train_i, test_i) = sss.split(dataframe, labels)



train, test = [dataframe[i] for i in train_i], [dataframe[i] for i in test_i]
train_labels, test_labels = [labels[i] for i in train_i], [labels[i] for i in test_i] 

artifact = wandb.Artifact("input_tables", type="split_data") 

with TemporaryFile() as temp_file: 
    np.save(temp_file,train)
    artifact.add_file(temp_file.get_temp_dir(), name="train") 
with TemporaryFile() as temp_file: 
    np.save(temp_file,test)
    artifact.add_file(temp_file.get_temp_dir(), name="test")    
with TemporaryFile() as temp_file: 
    np.save(temp_file,validation)
    artifact.add_file(temp_file.get_temp_dir(), name="validation")   
with TemporaryFile() as temp_file: 
    np.save(temp_file,train_labels)
    artifact.add_file(temp_file.get_temp_dir(), name="train") 
with TemporaryFile() as temp_file: 
    np.save(temp_file,test_labels)
    artifact.add_file(temp_file.get_temp_dir(), name="test")    
with TemporaryFile() as temp_file: 
    np.save(temp_file,validation_labels)
    artifact.add_file(temp_file.get_temp_dir(), name="validation") 
wandb.run.log_artifact(artifact) 

fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)
