import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from datetime import datetime
from sdae import StackedDenoisingAE

FILE_NAME = "GEO_data_batch_corr_final"

N_LAYERS = 3
N_NODES = [2000, 1000, 100]
DROPOUT = [0.1]
BATCH_SIZE = 2
EPOCHS = 10
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

    model, data, recon_mse = model.get_pretrained_sda(x_train, x_test, x_test, dir_out=output)

    """
        for i, layer in enumerate(model):
        w = layer.get_weights()
        print(layer.name)
        for j, arr in enumerate(w):
            print(arr.shape)
            np.savetxt('output/weights_{}{}_{}.txt'.format(i, j, arr.shape), arr)
    """

if __name__ == '__main__':

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_fit(os.path.join('output', now))
