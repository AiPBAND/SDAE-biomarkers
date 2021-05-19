from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime as dt
from clearml import Task
task = Task.init(project_name="sdae", task_name="preprocess")

DATA_SET = "gene_exp_MA.csv"
RANDOM_STATE = 20202
DATA_DIR = 'data'

run_id = dt.strftime(dt.now(),"%y%j%H%M")
DATA_DIR = os.path.join(DATA_DIR, run_id)

# create pandas dataframe from downloaded data
print("Preparing data for training")
raw_df = pd.read_csv(DATA_SET, header=0, sep=",", skipinitialspace=True, index_col=0)

# convert diagnosis attribute to binary format
#raw_df['class_gbm_1'] = raw_df['class_gbm_1'].map({'1': 1, '0': 0})

# Create train/test split
print("Saving training and test data sets")

train_df, test_df = train_test_split(raw_df, train_size=0.8, random_state=RANDOM_STATE)
if not os.path.isdir(DATA_DIR): os.makedirs(DATA_DIR)
train_df.to_pickle(os.path.join(DATA_DIR, 'train'))
test_df.to_pickle(os.path.join(DATA_DIR, 'test'))
    
print("Completed data preparation")

