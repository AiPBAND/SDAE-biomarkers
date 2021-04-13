import tensorflow as tf
print(tf.__version__)
import pandas as pd
import numpy as np
from src.models import EncoderStack
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import dataset
import os
import logging

import wandb
from wandb.keras import WandbCallback

wandb.login()

wandb.init(project='bio-tuning',
           entity='aipband',
           save_code=True,
           group="supervised",
           magic=True,

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-path",
    type=str,
    dest="data_path",
    help="Path to the CSV input data. First row headers. First column IDs",
    default="/data/GEO_features.csv",
)

parser.add_argument(
    "--batch",
    default=5,
    dest="BATCH_SIZE",
    type=int,
    help="Number of samples per batch.",
)
parser.add_argument(
    "--epochs", 
    default=10, 
    dest="EPOCHS",
     type=int, 
     help="Number of epochs."
)

config = wandb.config
args = parser.parse_args()
wandb.config.update(args)

dataframe = pd.read_csv(args.DATA_PATH, index_col=0)
data = dataframe.values[1:]
data = normalize(data)

x = datframe[:-2]
y =  to_categorical(datframe[-2])
cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())

wrap = lambda x,y = EncoderStack(encoder_models, 'output/').fit(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, num_epochs=EPOCHS)
scores = cross_validate(wrap, X, y, cv=3, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)

wandb.log('cross_val_scores', scores)
wandb.log('avg_score', mean(scores))
wandb.log('min_score', min(scores))
wandb.log('max_score', max(scores))
wandb.save("mymodel.h5")