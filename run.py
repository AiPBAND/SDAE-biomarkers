import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from datetime import datetime
from sdae import StackedDenoisingAE

FILE_NAME = "GEO_data_batch_corr_final"
OUTPUT_DIR = 'output/'

N_LAYERS = 2
N_NODES = [1000, 100]
DROPOUT = [0.1]

EPOCHS = 5
TEST_RATIO = 0.1

assert len(N_NODES) == N_LAYERS or len(N_NODES) == 1
assert len(DROPOUT) == N_LAYERS or len(DROPOUT) == 1

OUTPUT_DIR = os.path.join(OUTPUT_DIR, str(datetime.now()).replace(":", "."))
os.makedirs(OUTPUT_DIR)

dataframe = pd.read_pickle('data/pd/'+FILE_NAME)

data = dataframe.values
data = normalize(data)

X_train, X_test = train_test_split(data, test_size=TEST_RATIO)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = StackedDenoisingAE(n_layers=N_LAYERS, n_hid=N_NODES, dropout=DROPOUT, nb_epoch=EPOCHS)

model, data, recon_mse = model.get_pretrained_sda(X_train, X_test, X_test, dir_out=OUTPUT_DIR)

for i, layer in enumerate(model):
    w = layer.get_weights()
    print(layer.name)
    for j, arr in enumerate(w):
        print(arr.shape)
        np.savetxt('output/weights_{}{}_{}.txt'.format(i, j, arr.shape), arr)


