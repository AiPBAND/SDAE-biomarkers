from sdae import StackedDenoisingAE
from sklearn.preprocessing import normalize

train_file_path = "GSE4290_series_matrix.txt"

from numpy import genfromtxt
my_data = genfromtxt(train_file_path, delimiter='\t', skip_header=True)[:,1:]
my_data = normalize(my_data)

X_train = my_data[:1000,1:]
X_test = my_data[1000:,1:]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train_50 = X_train[0:50]

n_in = n_out = X_train.shape[1];
n_hid = 576

cur_sdae = StackedDenoisingAE(n_layers = 3, n_hid = [400], dropout = [0.1], nb_epoch = 10)
recon_mse = cur_sdae.get_pretrained_sda(X_train, X_test, X_test, dir_out = '../output/')
cur_sdae.save_weights('../output/weights')


