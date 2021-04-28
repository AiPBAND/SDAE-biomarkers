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
from wandb.keras import WandbCallback


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

    def fit(self, x_train, x_test, batch_size, num_epochs, loss_fn='mse', metrics
            optimizer='rmsprop', verbose=1, patience=1, cb=None, validation_data=None):

        self.set_output_path()

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)

        self.autoencoder_model.compile(loss=loss_fn,
                                        metrics=[], 
                                        optimizer=optimizer)

        self.autoencoder_model.fit(x_train, x_train,
            callbacks=[early_stop, self.tsb, WandbCallback()], epochs=num_epochs,
            batch_size=batch_size, validation_data=validation_data)

        

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
            update_freq='step')

    def _bce(self, x, y):
        pred = self.model.predict(x)
        return self.bce_func(y, pred).numpy()

    def get(self):

        self.set_output_path()
        self.name = "autoencoder-{}".format(self.num_hidden)

        return self.model