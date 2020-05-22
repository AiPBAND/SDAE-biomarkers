import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from datetime import datetime
from sdae import StackedDenoisingAE
import numpy as np
from matplotlib import pyplot as plt

FILE_NAME = "GEO_data_batch_corr_final"

N_LAYERS = 3
N_NODES = [1000, 500, 100]
DROPOUT = [0.1]
BATCH_SIZE = 3
EPOCHS = 3
TEST_RATIO = 0.15


def run_fit(output):

    assert len(N_NODES) == N_LAYERS or len(N_NODES) == 1
    assert len(DROPOUT) == N_LAYERS or len(DROPOUT) == 1

    dataframe = pd.read_pickle('data/pd/'+FILE_NAME)

    data = dataframe.values
    data = normalize(data)

    x_train, x_test = train_test_split(data, test_size=TEST_RATIO)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = StackedDenoisingAE(n_layers=N_LAYERS, n_hid=N_NODES, dropout=DROPOUT, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    encoders, data, recon_mse = model.get_pretrained_sda(x_train, x_test, x_test, dir_out=output)

    weights = []
    for level in encoders:
        weights.append(level.get_weights()[0])
    influence = np.linalg.multi_dot(weights)
    print(influence.shape)

    plt.imshow(influence[:500])
    plt.show()
    total = np.sum(influence,1)
    plt.bar(range(len(total)), total)
    plt.show()

if __name__ == '__main__':

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_fit(os.path.join('output', now))
