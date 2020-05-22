"""
@author madhumita
"""
import numpy as np
import os
import nn_utils
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K


class StackedDenoisingAE(object):
    """
    Implements stacked denoising autoencoders in Keras, without tied weights.
    To read up about the stacked denoising autoencoder, check the following paper:
    
    Vincent, Pascal, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 
    "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." 
    Journal of Machine Learning Research 11, no. Dec (2010): 3371-3408.
    """

    def __init__(self, n_layers=1, n_hid=[500], dropout=[0.05], enc_act=['sigmoid'], dec_act=['linear'], bias=True,
                 loss_fn='mse', batch_size=32, nb_epoch=300, optimizer='rmsprop', verbose=1):
        """
        Initializes parameters for stacked denoising autoencoders
        @param n_layers: number of layers, i.e., number of autoencoders to stack on top of each other.
        @param n_hid: list with the number of hidden nodes per layer. If only one value specified, same value is used for all the layers
        @param dropout: list with the proportion of data_in nodes to mask at each layer. If only one value is provided, all the layers share the value.
        @param enc_act: list with activation function for encoders at each layer. Typically sigmoid.
        @param dec_act: list with activation function for decoders at each layer. Typically the same as encoder for binary data_in, linear for real data_in.
        @param bias: True to use bias value.
        @param loss_fn: The loss function. Typically 'mse' is used for real values. Options can be found here: https://keras.io/objectives/ 
        @param batch_size: mini batch size for gradient update
        @param nb_epoch: number of epochs to train each layer for
        @param optimizer: The optimizer to use. Options can be found here: https://keras.io/optimizers/
        @param verbose: 1 to be verbose
        """
        self.n_layers = n_layers
        self.bias = bias
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_hid, self.dropout, self.enc_act, self.dec_act = nn_utils.assert_input(n_layers, n_hid, dropout, enc_act, dec_act)

    def get_pretrained_sda(self, data_in, data_val, data_test, dir_out, model_layers=None):
        """
        Pretrains layers of a stacked denoising autoencoder to generate low-dimensional representation of data.
        Returns a Sequential model with the Dropout layer and pretrained encoding layers added sequentially. 
        Optionally, we can return a list of pretrained sdae models by setting get_enc_model to False. 
        Additionally, returns dense representation of input, validation and test data. 
        This dense representation is the value of the hidden node of the last layer.
        The cur_model be used in supervised task by adding a classification/regression layer on top, 
        or the dense pretrained data can be used as input of another cur_model.
        @param data_in: input data (scipy sparse matrix supported)
        @param data_val: validation data (scipy sparse matrix supported)
        @param data_test: test data (scipy sparse matrix supported)
        @param dir_out: output directory to write cur_model
        @param get_enc_model: True to get a Sequential model with Dropout and encoding layers from SDAE. 
                              If False, returns a list of all the encoding-decoding models within our stacked denoising autoencoder.
        @param write_model: True to write cur_model to file
        @param model_layers: Pretrained cur_model layers, to continue training pretrained model_layers, if required
        """

        if model_layers is not None:
            self.n_layers = len(model_layers)
        else:
            model_layers = [None] * self.n_layers

        encoders = []

        recon_mse = 0

        for cur_layer in range(self.n_layers):

            if model_layers[cur_layer] is None:
                input_layer = Input(shape=(data_in.shape[1],))

                # masking input data to learn to generalize, and prevent identity learning
                dropout_layer = Dropout(rate=self.dropout[cur_layer])
                in_dropout = dropout_layer(input_layer)

                encoder_layer = Dense(units=self.n_hid[cur_layer],
                                      kernel_initializer='glorot_uniform',
                                      activation=self.enc_act[cur_layer],
                                      name='encoder' + str(cur_layer),
                                      use_bias=self.bias)
                encoder = encoder_layer(in_dropout)

                decoder_layer = Dense(units=data_in.shape[1],
                                      kernel_initializer='glorot_uniform',
                                      activation=self.dec_act[cur_layer],
                                      name='decoder' + str(cur_layer),
                                      use_bias=self.bias)
                decoder = decoder_layer(encoder)

                cur_model = Model(input_layer, decoder)

                cur_model.compile(loss=self.loss_fn, optimizer=self.optimizer)

            else:
                cur_model = model_layers[cur_layer]

            print("Training layer " + str(cur_layer))

            output_folder = dir_out+"-"+str(cur_layer)
            tensorboard = TensorBoard(log_dir=output_folder,
                                      write_graph=True,
                                      update_freq='batch')

            # Early stopping to stop training when val loss increases for 1 epoch
            early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0)

            cur_model.fit(x=data_in,
                          y=data_in,
                          callbacks=[early_stopping, tensorboard],
                          epochs=self.nb_epoch,
                          batch_size=self.batch_size,
                          shuffle=True,
                          validation_data=(data_val, data_val))  # does not affect training

            print("Layer " + str(cur_layer) + " has been trained")

            model_layers[cur_layer] = cur_model
            cur_model.save(os.path.join(output_folder, 'model'))
            encoders.append(cur_model.layers[-2])

            if cur_layer == 0:
                recon_mse = self._get_recon_error(cur_model, data_in, n_out=cur_model.layers[-1].output_shape[1])

            # train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, 
            # used only to learn weights. output of second layer: hidden layer
            data_in = self._get_intermediate_output(cur_model, data_in, n_layer=2, train=0, 
                                                    n_out=self.n_hid[cur_layer], batch_size=self.batch_size)  
            
            assert data_in.shape[1] == self.n_hid[cur_layer], "Output of hidden layer not retrieved"

            # get output of second layer (hidden layer) without dropout
            data_val = self._get_intermediate_output(cur_model, data_val, n_layer=2, train=0, 
                                                     n_out=self.n_hid[cur_layer], batch_size=self.batch_size)
            
            data_test = self._get_intermediate_output(cur_model, data_test, n_layer=2, train=0,
                                                      n_out=self.n_hid[cur_layer], batch_size=self.batch_size)

        self._write_sda_config(dir_out)

        return encoders, (data_in, data_val, data_test), recon_mse

    def _write_sda_config(self, dir_out):
        """
        Write the configuration of the autoencoder to a file
        @param cur_sdae: autoencoder class
        @param cfg: config object
        """
        with open(dir_out + 'sdae_config.txt', 'w+') as f:
            f.write("Number of layers: " + str(self.n_layers))
            f.write("\nHidden nodes: ")
            for i in range(self.n_layers):
                f.write(str(self.n_hid[i]) + ' ')

            f.write("\nDropout: ")
            for i in range(self.n_layers):
                f.write(str(self.dropout[i]) + ' ')

            f.write("\nEncoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.enc_act[i]) + ' ')

            f.write("\nDecoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.dec_act[i]) + ' ')

            f.write("\nEpochs: " + str(self.nb_epoch))

            f.write("\nBias: " + str(self.bias))
            f.write("\nLoss: " + str(self.loss_fn))
            f.write("\nBatch size: " + str(self.batch_size))
            f.write("\nOptimizer: " + str(self.optimizer))

    def _get_intermediate_output(self, model, data_in, n_layer, train, n_out, batch_size, dtype=np.float32):
        """
        Returns output of a given intermediate layer in a model
        @param model: model to get output from
        @param data_in: sparse representation of input data
        @param n_layer: the layer number for which output is required
        @param train: (0/1) 1 to use training config, like dropout noise.
        @param n_out: number of output nodes in the given layer (pre-specify so as to use generator function with sparse matrix to get layer output)
        @param batch_size: the num of instances to convert to dense at a time
        @return value of intermediate layer
        """
        data_out = np.zeros(shape=(data_in.shape[0], n_out))

        x_batch_gen = nn_utils.x_generator(data_in, batch_size=batch_size, shuffle=False)
        stop_iter = int(np.ceil(data_in.shape[0] / batch_size))

        for i in range(stop_iter):
            cur_batch, cur_batch_idx = next(x_batch_gen)
            data_out[cur_batch_idx, :] = self._get_nth_layer_output(model, n_layer, X=cur_batch, train=train)

        return data_out.astype(dtype, copy=False)

    def _get_nth_layer_output(self, model, n_layer, X, train=1):
        """
        Returns output of nth layer in a given model.
        @param model: keras model to get an intermediate value out of
        @param n_layer: the layer number to get the value of
        @param X: input data for which layer value should be computed and returned.
        @param train: (1/0): 1 to use the same setting as training (for example, with Dropout, etc.), 0 to use the same
        setting as testing phase for the model.
        @return the value of n_layer in the given model, input, and setting 
        """
        get_nth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                          [model.layers[n_layer].output])
        return get_nth_layer_output([X, train])[0]

    def _get_recon_error(self, model, data_in, n_out):
        """
        Return reconstruction squared error at individual nodes, averaged across all instances.
        @param model: trained model
        @param data_in: input data to reconstruct
        @param n_out: number of model output nodes
        """
        train_recon = self._get_intermediate_output(model, data_in, n_layer=-1, train=0, n_out=n_out,
                                                    batch_size=self.batch_size)

        recon_mse = np.mean(np.square(train_recon - data_in), axis=0)
        recon_mse = np.ravel(recon_mse)

        return recon_mse
