import os
import numpy as np
import pandas as pd
import logging
# import signal
import h5py
import datetime
# from datetime import datetime
import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras.backend as K
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras import losses
# from tensorflow.keras import models
# from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import load_model, save_model
# from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
from tensorflow.keras import Sequential as ks
import tensorflow.keras.regularizers as kr
import tensorflow.keras.initializers as ki
from tensorflow.keras import optimizers as ko
# import tensorflow.python.keras.optimizer_v2 as ko
import tensorflow.keras.activations as ka



class AE(object):
    def __init__(self, input_dim=4096, latent_dim=32, hidden_units=[], reg1=None, encoder_dp=0., loss='mae', lr=0.01):
        # super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = int(latent_dim)
        self.hidden_units = np.array(hidden_units)
        self.units = self.get_units()

        self.reg1 = reg1
        self.encoder_dp=encoder_dp

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        self.opt = ko.Adam(learning_rate = lr, decay = 1e-6)
        self.loss = loss
        self.log_dir =  "logs/fit/" + self.loss + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # self.opt = ko.SGD(learning_rate=0.001, momentum=0.9)
        self.ae = self.get_ae()

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6),
            tf.keras.callbacks.ReduceLROnPlateau('loss', 
                                                patience=3, 
                                                min_lr=0., 
                                                factor=0.1),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        ]



    def fit(self, x_train, ep=50):
        self.ae.fit(x_train, x_train, 
                    epochs=ep, 
                    batch_size=16, 
                    validation_split=0.1, 
                    callbacks=self.callbacks,
                    shuffle=True
                    )


    def set_model_shapes(self, input_shape, latent_size):
        self.input_shape = (input_shape[1], )
        self.latent_size = latent_size

    def get_units(self):
        self.hidden_units = self.hidden_units[self.hidden_units > self.latent_dim]
        units = [self.input_dim, *self.hidden_units, self.latent_dim]
        print(units)
        return units 

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_ae(self):
        ae = keras.Model(self.encoder_input, self.call(self.encoder_input), name="ae")
        ae.summary()
        ae.compile(self.opt, loss = self.loss)
        return ae

    def get_encoder(self):
        # x = kl.Flatten()(encoder_input) #dense layer\
        # x = kl.Dense(self.hidden_units[0])(self.encoder_input)
        # x = kl.BatchNormalization()(x)
        # x = kl.LeakyReLU()(x)
        x = self.encoder_input
        # x = kl.Dense(self.input_dim, name='first')(x)
        # x = kl.Dense(self.input_dim//2, activation = ka.linear)(x)
        for ii, unit in enumerate(self.units[1:]):
        # for ii, unit in enumerate(self.units[2:]):
            name = 'encod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1, name=name)(x)
        # x = kl.Dense(self.latent_dim)(x)
        encoder = keras.Model(self.encoder_input, x, name="encoder")
        # encoder.summary()
        return encoder

    def get_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        # x = kl.Dense(self.hidden_units[-1])(latent_input)
        # x = kl.LeakyReLU()(x)
        x = latent_input
        # for ii, unit in enumerate(self.units[-2::-1]):
        for ii, unit in enumerate(self.hidden_units[::-1]):
            name = 'decod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1, name=name)(x)
        x = kl.Dense(self.input_dim, name='last')(x)
        decoder = keras.Model(latent_input, x, name="decoder")
        # decoder.summary()
        return decoder

    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
        # layer = ks([kl.Dense(unit, activation='tanh')
        if reg1 is not None:
            kl1 = tf.keras.regularizers.l1(reg1)
        else:
            kl1 = None

        layer = ks([kl.Dense(unit, kernel_regularizer=kl1, name=name),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(dp_rate)
                    # keras.activations.tanh()
                    ])
        return layer


# n = 4096
# encoder_input = keras.Input(shape = n, name = 'img')
# x = keras.layers.Flatten()(encoder_input) #dense layer\
# x = keras.layers.Dense(n//2, activation="relu")(x)
# x = keras.layers.Dense(n//8, activation="relu")(x)
# # x = keras.layers.Dense(n//8, activation="relu")(x)
# encoder_output = x
# encoder = keras.Model(encoder_input, encoder_output, name="encoder")

# decoder_input = keras.layers.Dense(n//8, activation="relu")(encoder_output)
# x = keras.layers.Dense(n//8, activation="relu")(decoder_input)
# x = keras.layers.Dense(n//2, activation="relu")(x)
# x = keras.layers.Dense(n, activation="relu")(x)
# decoder_output = x

# # decoder_output = keras.layers.Reshape((28,28,1))(decoder_input)
# opt = keras.optimizers.Adam(learning_rate = 0.001, decay = 1e-6)

# autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
# autoencoder.summary()

