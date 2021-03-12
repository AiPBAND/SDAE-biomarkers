# %%
"""
# supervised training of the encoder layers
"""

# %%
"""
## Initial setup
### Imports
"""

# %%
import tensorflow as tf
print(tf.__version__)
import pandas as pd
import numpy as np
from ../src/models import EncoderStack
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import normalize
import dataset
import neptune
import neptune_tensorboard as neptune_tb
import os
import logging

# %%


# %%
"""
### Set experiment configuration
"""

# %%
config = {
    "N_NODES": [1000, 500, 100],
    "DROPOUT": [0.1],
    "BATCH_SIZE": 15,
    "EPOCHS": 5,
    "TEST_RATIO": 0.30,
    "DATA_BUCKET": "sdae-geo",
    "DATA_OBJECT": "GEO_data_batch_corr_final.csv",
    "DATA_LABELS": " GBM_class.csv",
    "VERBOSITY": 2,
    "LOG_DIR": "./log_dir",
    "PATIENCE":3
}


# %%
"""
## Initialize Netptune and Tensorboard logging
"""

# %%
os.environ['NEPTUNE_API_TOKEN']="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjNiYmZhYjEtNzc3ZS00Y2NhLWI5NTgtYWU0MmQyMWJhM2I0In0="
os.environ['NEPTUNE_PROJECT']="jgeof/sdae"
os.environ['NEPTUNE_NOTEBOOK_ID']="ecd86e96-4da7-44e3-9a17-43da2dfcae35"
os.environ['NEPTUNE_NOTEBOOK_PATH']="constrained-SDAE/sdae.ipynb"

neptune.init(os.environ['NEPTUNE_PROJECT'], api_token=os.environ['NEPTUNE_API_TOKEN'])

logger = logging.getLogger("SDAE")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

experiment = neptune.create_experiment(name='configuration', params=config, logger=logger)

os.environ['EXP_DIR'] = os.path.join(config["LOG_DIR"], experiment.id)
os.mkdir(os.environ['EXP_DIR'])

logger.info("project directory: {}".format(os.environ['EXP_DIR']))
!neptune tensorboard ${EXP_DIR} --project ${NEPTUNE_PROJECT}
%load_ext tensorboard

# %%
"""
### Start tensorboard server
Tensorboard by running the following command in a terminal:
"""

# %%
print("tensorboard --logdir {} --bind_all".format(os.environ['EXP_DIR']))

# %%
"""
**Tensorboard cannot server over HTTPS, use external HTTP url: http://34.77.45.86:6006/**
"""

# %%
"""
# Load and preprocess the data
"""

# %%
"""
## Load data from Google Storage
"""

# %%
dataframe = dataset.load_gs_data(config['DATA_BUCKET'], config['DATA_OBJECT'], os.environ['EXP_DIR'])

# %%
classes = pd.read_csv('data/pd/class.csv', header=None, index_col=0).values
classes = to_categorical(classes)


model = EncoderStack(encoder_models, 'output/')

print("\n##################################################################")
print("Training layer {} with {} hidden nodes..\n".format(idx, num_hidden))
loss_train, loss_test = model.fit(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, num_epochs=EPOCHS)

print("\nTraining losss: ", loss_train)
print("\nTesting loss: ", loss_test)