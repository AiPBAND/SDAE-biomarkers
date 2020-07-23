import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#from tensorflow.keras.utils import to_categorical
from datetime import datetime
from models.dae import AutoencoderStack
import numpy as np
from matplotlib import pyplot as plt

FILE_NAME = "GEO_data_batch_corr_final"

N_LAYERS = 2
N_NODES = [100, 50]
DROPOUT = [0.1]
BATCH_SIZE = 10
EPOCHS =1
TEST_RATIO = 0.15


def run_fit(output):

    assert len(N_NODES) == N_LAYERS or len(N_NODES) == 1
    assert len(DROPOUT) == N_LAYERS or len(DROPOUT) == 1

    dataframe = pd.read_pickle('data/pd/'+FILE_NAME)

    print(dataframe.shape)

    data = dataframe.values
    data = normalize(data)

    classes = np.random.randint(0,3, dataframe.shape[0])
    classes = to_categorical(classes)

    x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=TEST_RATIO)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = AutoencoderStack(x_train.shape[1], num_stacks=N_LAYERS, hidden_nodes=N_NODES)

    encoders, data, recon_mse = model.unsupervised_fit(x_train, x_test, x_test, dir_out=output)

    #fit_classifier = model.supervised_classification(amodel=encoders, x_train=x_train, x_val=x_test, y_train=y_train, y_val=y_test, n_classes=3, dir_out=output)
# pred = cur_sdae.predict(fit_classifier, X_test, 1337)
# score, conf_matrix, error_idx = model_utils.score(y_true, y_pred, y_pred_score, cfg, n_classes)

if __name__ == '__main__':

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_fit(os.path.join('output', now))
