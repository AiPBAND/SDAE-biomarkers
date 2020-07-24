"""
@author madhumita
"""
import numpy as np
import os
import nn_utils
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError

class StackedDenoisingAE(object):
    """
    Implements stacked denoising autoencoders in Keras, without tied weights.
    To read up about the stacked denoising autoencoder, check the following paper:

    Vincent, Pascal, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol.
    "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."
    Journal of Machine Learning Research 11, no. Dec (2010): 3371-3408.
    """

    def __init__(self, n_layers=1, n_hid=[500], dropout=[0.05], enc_act=['sigmoid'],
        dec_act=['linear'], bias=True, loss_fn='mse', batch_size=32, nb_epoch=300,
        optimizer='rmsprop', verbose=1):
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
        self.n_hid, self.dropout, self.enc_act, self.dec_act = nn_utils.assert_input(n_layers,
            n_hid, dropout, enc_act, dec_act)
        self.mse = MeanSquaredError()
        self.params = [attr for attr in dir(self) if not
            callable(getattr(self, attr)) and not attr.startswith("__")]

    def get_pretrained_sda(self, data_in, data_val, data_test, dir_out, get_enc_model=True,
        write_model=True, model_layers=None, dropout_all=False,final_act_fn='softmax'):
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

        num_features = data_in.shape[1]

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
            print(cur_model.summary())
            cur_model.fit(x=data_in, y=data_in, callbacks=[early_stopping, tensorboard], epochs=self.nb_epoch,
                batch_size=self.batch_size, shuffle=True, validation_data=(data_val, data_val))

            print("Layer " + str(cur_layer) + " has been trained")

            model_layers[cur_layer] = cur_model
            cur_model.save(os.path.join(output_folder, 'model'))
            encoders.append(cur_model.layers[-2])

            if cur_layer == 0:
                recon_mse = self._get_mse(data_in, cur_model.layers[-1].output)

            temp_model = Model(inputs=input_layer, outputs=encoder)
            data_in = temp_model.predict(data_in)

            assert data_in.shape[1] == self.n_hid[cur_layer], "Output of hidden layer not retrieved"

            data_val = temp_model.predict(data_val)
            data_test = temp_model.predict(data_test)

        self._write_sda_config(dir_out)

        # Build encoder model ##################################################
        final_input = Input(shape=(num_features,))
        final_dropout = Dropout(self.dropout[0])
        final_in_drop = final_dropout(final_input)

        last_layer = [final_in_drop]
        for i in range(len(encoders)):
            if i and dropout_all:
                dropout_layer = Dropout(self.dropout[i])
                last_layer = dropout_layer(last_layer)
            last_layer.append(encoders[i](last_layer[-1]))
        self.last_layer = last_layer[-1]
        self.final_input = final_input
        final_model = Model(final_input, last_layer[-1])

        if write_model:
            nn_utils.save_model(final_model, out_dir=dir_out, f_arch='enc_layers.png',
                f_model='enc_layers.json', f_weights='enc_layers_weights.h5')

        return final_model, (data_in, data_val, data_test), recon_mse

    def supervised_classification(self, amodel, x_train, y_train, x_val, y_val,
        n_classes, dir_out, x_test=None, y_test=None, final_act_fn='softmax',
        loss='categorical_crossentropy', get_recon_error=False):
        """
        Classification by finetuning a pretrained autoencoder model for a
        given task
        """
        # Build classification model ###########################################
        dense_class_out = Dense(n_classes, activation=final_act_fn)
        dense_out = dense_class_out(amodel.output)
        model = Model(amodel.inputs, dense_out)


        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model.png')


        model.compile(loss=loss, optimizer=self.optimizer)
        self._write_sda_config(dir_out)

        early_stopping = EarlyStopping(monitor='val_loss', patience=1,
        verbose=self.verbose)

        model.fit(x=x_train, callbacks=[early_stopping], epochs=self.nb_epoch,
            batch_size=self.batch_size, shuffle=True, verbose=self.verbose,
            validation_data=(x_val, y_val))
        print(model.summary())
        extractor = self._get_extractor(model)

        final_train = extractor(data_in)[-2]
        final_val = extractor(data_val)[-2]
        final_test = extractor(data_test)[-2] if data_test else None

        recon_mse = self._get_mse(data_in, final_train)

        return model, (final_train, final_val, final_test), recon_mse

    def _write_sda_config(self, dir_out):
        """
        Write the configuration of the autoencoder to a file
        """
        with open(dir_out + 'sdae_config.txt', 'w+') as f:
            for param in self.params:
                f.writelines("{}: {}\n".format(param,
                self.__getattribute__(param)))

    def _get_extractor(self, model):
        layers = [layer.output for layer in model.layers]
        return Model(inputs=model.inputs, outputs=layers)

    def _get_mse(self, data_true, data_pred):
        return self.mse(data_true, data_pred)
