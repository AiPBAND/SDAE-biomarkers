from sdae import StackedDenoisingAE
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import numpy as np
import os, shutil
my_data = np.load('data/output_filter/data.npy')
my_data = normalize(my_data)

X_train, X_test = train_test_split(my_data, test_size=0.1)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if os.path.exists("output"):
    shutil.rmtree("output")

os.mkdir('output')

cur_sdae = StackedDenoisingAE(n_layers=2, n_hid=[1000, 100], dropout=[0.1], nb_epoch=5)
model, data, recon_mse = cur_sdae.get_pretrained_sda(X_train, X_test, X_test, dir_out='output/', get_enc_model=False)

for i, layer in enumerate(model):
    w = layer.get_weights()
    print(layer.name)
    for j, arr in enumerate(w):
        print(arr.shape)
        np.savetxt('output/weights_{}{}_{}.txt'.format(i, j, arr.shape), arr)


