import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras import Sequential as ks
import tensorflow.keras.regularizers as kr
import tensorflow.keras.initializers as ki
from tensorflow.keras import optimizers as ko
import tensorflow.keras.activations as ka

class DumbAE(object):
    def __init__(self, input_dim=4096, latent_dim=4096, encoder_dp=0., reg1=None, lr=0.001):
        # super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = latent_dim
        self.hidden_units = np.array([])
        self.units = self.get_units()

        self.reg1 = reg1
        self.encoder_dp = encoder_dp

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()


        self.opt = ko.Adam(learning_rate = lr, decay = 1e-6)
        self.loss = 'mae'
        # self.loss = 'mse'
        # self.opt = ko.SGD(learning_rate=0.001, momentum=0.9)
        self.ae = self.get_ae()

        self.callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
            tf.keras.callbacks.ReduceLROnPlateau('loss', patience=3, min_lr=0., factor=0.1)
        ]


    def fit(self, x_train, ep=50):
        self.ae.fit(x_train, x_train, 
                    epochs=ep, 
                    batch_size=16, 
                    validation_split=0.1, 
                    callbacks=self.callbacks,
                    shuffle=True
                    )


    # def set_model_shapes(self, input_shape, latent_dim):
    #     self.input_dim = (input_shape[1], )

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
        # x = kl.Dense(self.latent_dim, activation = ka.linear)(x)
        for ii, unit in enumerate(self.units[1:]):
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1)(x)
        encoder = keras.Model(self.encoder_input, x, name="encoder")
        # encoder.summary()
        return encoder

    def get_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        # x = kl.Dense(self.hidden_units[-1])(latent_input)
        # x = kl.LeakyReLU()(x)
        x = latent_input
        for ii, unit in enumerate(self.units[-2::-1]):
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp,)(x)
        # x = kl.Dense(self.input_dim, activation=ka.linear)(x)
        
        decoder = keras.Model(latent_input, x, name="decoder")
        # decoder.summary()
        return decoder

    def add_dense_layer(self, unit, dp_rate=0.0, reg1=None):
        if reg1 is not None:
            kl1 = tf.keras.regularizers.l1(reg1)
        else:
            kl1 = None
        layer = ks([kl.Dense(unit, kernel_regularizer=kl1),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(dp_rate)
                    ])
        return layer
