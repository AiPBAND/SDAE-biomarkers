import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime
import os

class DenoisingAutoencoder(object):

    def __init__(self, inputs, output_path, num_hidden=500, dropout_rate=0.05,
        encoder_act='sigmoid', decoder_act='linear', bias=True, loss_fn='mse',
        batch_size=32, num_epochs=300, optimizer='rmsprop', verbose=1):

        self.inputs = inputs
        self.num_hidden = num_hidden
        self.dropout = dropout_rate
        self.encoder_act = encoder_act
        self.decoder_act = decoder_act
        self.bias = bias
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.verbose = verbose

        self.num_input = inputs.shape[1]

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log = os.path.join(now, output_path)

        self.tsb = TensorBoard(log_dir=self.log, write_graph=True,
            update_freq='batch')

        self.mse_obj = MeanSquaredError()

        # Build layers #########################################################

        dropout_layer = Dropout(rate=self.dropout_rate)
        dropout_output = dropout_layer(self.inputs)

        self.encoder_layer = Dense(units=self.num_hidden,
            kernel_initializer='glorot_uniform', activation=self.encoder_act,
            name="encoder {}->{}".format(num_inputs, num_hidden),
            use_bias=self.bias)
        self.encoder_ouput = self.encoder_layer(dropout_output)

        self.decoder_layer = Dense(units=inputs.shape[1],
            kernel_initializer='glorot_uniform', activation=self.decoder_act,
            name="decdoer {}->{}".format(num_hidden, num_inputs),
            use_bias=self.bias)
        self.decoder_ouput = self.decoder_layer(encoder_ouput)

        # Build model ##########################################################

        self.autoencoder_model = Model(self.inputs, self.decoder_ouput)
        self.autoencoder_model.compile(loss=self.loss_fn, optimizer=self.optimizer)

        self.encoder_model(self.inputs, self.encoder_ouput)


    @property
    def encoder_output(self):
        return self.encoder_ouput

    @property
    def decoder_output(sefl):
        return self.decoder_ouput

    @property
    def encoder_layer(slef):
        return self.encoder_layer

    @property
    def decoder_layer(self):
        return self.decoder_layer

    @property
    def autoencoder_model(self):
        return self.autoencoder

    @property
    def encoder_model(self):
        return self.encoder_model

    def _mse(self, real):
        recon = self.autoencoder_model.predict(real)
        return self.mse_obj(real, recon)

    def fit_unsupervised(self, data_train, data_val, data_test):

        early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=0)

        self.autoencoder_model.fit(x=data_train, y=data_train,
            callbacks=[early_stop, self.tsb], epochs=self.num_epochs,
            batch_size=self.batch_size, shuffle=True,
            validation_data=(data_val, data_val))

        self.autoencoder_model.save(os.path.join(self.log, 'model'))

        return self._mse(data_train), self._mse(data_val), self._mse(data_test)

class AutoencoderStack(object):

    def __init__(self, num_features, num_stacks=2, hidden_nodes=[500, 100], output_path="/"):

        assert num_stacks == len(hidden_nodes) or len(hidden_nodes) == 1

        self.num_features = num_features
        self.num_stacks = num_stacks
        self.hidden_nodes = hidden_nodes
        self.output_path = output_path

        self.stack = []
        self.inputs = Input(shape=(num_features,))

        for i in range(self.num_stacks):
            input_layer = self.inputs if not i else stack[-1].encoder_ouput()

            model = DenoisingAutoencoder(self.inputs, output_path=self.output_path, num_hidden=hidden_nodes[i])

            if i:
                stack[-1].decoder(model.decoder_output)

            self.stack.append(model)

    def unsupervised_fit(self, data_train, data_val, data_test, output_dir):

        mse_per_layer = []

        for layer in self.stack:
            mse_tuple = layer.fit_unsupervised(data_train, data_val, data_test)
            mse_per_layer.append(mse_tuple)

        return mse_per_layer



