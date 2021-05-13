from ludwig.api import LudwigModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml
from datetime import datetime as dt

DATA_SET = "gene_exp_MA.csv"
RANDOM_STATE = 20202
DATA_DIR = 'data'
EPOCHS = 50
BATCHES = 5
LATENT = [1000,500]


layers = []
run_id = dt.strftime(dt.now(),"%y%j%H%M")
for lx, num_latent_nodes in enumerate(LATENT):

    # create pandas dataframe from downloaded data
    print("Preparing data for training")
    raw_df = pd.read_csv(DATA_SET,
                        header=0,
                        sep=",", skipinitialspace=True)

    print(raw_df.head(2))

    raw_df.columns = ['SAMPLE_ID'] + ['X' + str(i) for i in range(1, raw_df.shape[1]-1)] + ['class_gbm_1']

    # convert diagnosis attribute to binary format
    raw_df['class_gbm_1'] = raw_df['class_gbm_1'].map({'1': 1, '0': 0})

    # Create train/test split
    print("Saving training and test data sets")
    layer_format = (raw_df.shape[1]-2 if lx == 0 else LATENT[lx-1], num_latent_nodes)
    train_df, test_df = train_test_split(raw_df, train_size=0.8, random_state=RANDOM_STATE)
    if not os.path.isdir(DATA_DIR): os.mkdir(DATA_DIR)
    train_df.to_csv(os.path.join(DATA_DIR, '{}-{}x{}train.csv'.format(run_id,*layer_format)), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, '{}-{}x{}test.csv'.format(run_id,*layer_format)), index=False)

    print("Preparing Ludwig config")
    # Create ludwig input_features
    num_features = ['X' + str(i) for i in range(1, raw_df.shape[1]-1)]
    input_features = []

    # setup input features for numerical variables
    for p in num_features:
        a_feature = {'name': p, 
                    'type': 'numerical',
                    'preprocessing': {'normalization': 'zscore'}}
        input_features.append(a_feature)

    # Create ludwig output features
    output_features = [
        {
            'name': 'embeding',
            'type': 'numerical',
            'num_fc_layers': 1,
            'fc_size': num_latent_nodes
        }
    ]

    filename = '{}-{}x{}config.yaml'.format(run_id,*layer_format)
    layers.append(filename)

    config = {
        'input_features': input_features,
        'output_features': input_features,
        'run_id' : run_id,
        'training': {
            'epochs': EPOCHS,
            'batch_size': BATCHES
        },
    }

    if os.path.isfile(filename):
        os.remove(filename)
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        yaml.dump(config, f)    
    # setup ludwig config

if os.path.isfile(run_id):
        os.remove(run_id)    
with open(os.path.join(DATA_DIR, run_id+".yaml"), 'w') as f:
        yaml.dump({"layers": layers}, f)     
print("Completed data preparation")

