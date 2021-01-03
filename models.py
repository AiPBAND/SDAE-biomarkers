import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import sigmoid
from datetime import datetime
import os
from typing import List

class Autoencoder:

    def __init__(self, num_features, num_hidden, output_dir, dropout_rate=0.05,
        encoder_act='sigmoid', decoder_act='linear'):

        self.output_dir = output_dir
        self.mse_func = MeanSquaredError()

        self.num_hidden = num_hidden
        self.num_inputs = num_features

        self.encoder_act = encoder_act

        input_layer = Input(num_features)
        dropout_layer = Dropout(dropout_rate)
        dropout_output = dropout_layer(input_layer)

        self.encoder_layer = Dense(units=self.num_hidden, activation=encoder_act)
        self.encoder_ouput = self.encoder_layer(dropout_output)

        self.decoder_layer = Dense(units=self.num_inputs, activation=decoder_act)
        self.decoder_ouput = self.decoder_layer(self.encoder_ouput)

        self.autoencoder_model = Model(input_layer, self.decoder_ouput)
        self.encoder_model = Model(input_layer, self.encoder_ouput)

    def _mse(self, real):
        recon = self.autoencoder_model.predict(real)
        return self.mse_func(real, recon).numpy()

    def set_output_path(self):
        self.tsb = TensorBoard(log_dir=self.output_dir, write_graph=True,
            update_freq='batch')

    def fit(self, x_train, x_test, batch_size, num_epochs, loss_fn='mse',
            optimizer='rmsprop', verbose=1, patience=1):

        self.set_output_path()

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)

        self.autoencoder_model.compile(loss=loss_fn, optimizer=optimizer)

        self.autoencoder_model.fit(x_train, x_train,
            callbacks=[early_stop, self.tsb], epochs=num_epochs,
            batch_size=batch_size, validation_data=(x_test, x_test))

        name = "autoencoder-{}".format(self.num_hidden)

        save_path = os.path.join(self.output_dir, name)

        self.autoencoder_model.save(save_path)
        print("Trained model saved at: {}".format(save_path))

        return self._mse(x_train), self._mse(x_test)

class EncoderStack:

    def __init__(self, autoencoders: List[Autoencoder], output_dir, num_classes=2,
        dropout_rate=0.05, activation='sigmoid'):

        self.output_dir = output_dir
        self.bce_func = BinaryCrossentropy()

        model = Sequential()
        model.add(Input(shape=autoencoders[0].num_inputs))

        for ae in autoencoders:
            if dropout_rate:
                model.add(Dropout(dropout_rate))
            dense = Dense(ae.num_hidden, activation=ae.encoder_act)

            model.add(dense)
            dense.set_weights(ae.encoder_layer.get_weights())

        model.add(Dense(num_classes, activation=activation))
        self.model = model

    def set_output_path(self):
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join(self.output_dir, now)
        self.log_path = os.path.join(self.output_path, "log")
        self.tsb = TensorBoard(log_dir=self.log_path, write_graph=True,
            update_freq='batch')

    def _bce(self, x, y):
        pred = self.model.predict(x)
        return self.bce_func(y, pred).numpy()

    def fit(self, x_train, y_train, x_test, y_test, batch_size, num_epochs,
        loss_fn='mse', optimizer='rmsprop', verbose=1):

        self.set_output_path()

        self.model.compile(loss=loss_fn, optimizer=optimizer)

        early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=2)

        self.model.fit(x_train, y_train, callbacks=[early_stop, self.tsb],
            epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))


        return self._bce(x_train, y_train), self._bce(x_test, y_test)