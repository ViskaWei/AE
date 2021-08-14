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

class DumbAEmse(object):
    def __init__(self, latent_dim=32):

        self.input_dim = 4096
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = latent_dim

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        self.opt = ko.Adam(learning_rate = 0.001, decay = 1e-6)
        self.ae = self.get_ae()
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=2, min_lr=0., factor=0.3)
        ]

    def fit(self, x_train, ep=50):
        self.ae.fit(x_train, x_train, 
                    epochs=ep, 
                    batch_size=16, 
                    validation_split=0.1, 
                    callbacks=self.callbacks,
                    shuffle=True
                    )

    def get_ae(self):
        encoded = self.encoder(self.encoder_input)
        decoded = self.decoder(encoded)
        ae = keras.Model(self.encoder_input, decoded, name="ae")
        ae.summary()
        ae.compile(self.opt, loss = 'mse')
        return ae

    def get_encoder(self):
        x = kl.Dense(self.latent_dim)(self.encoder_input)
        encoder = keras.Model(self.encoder_input, x, name="encoder")
        return encoder

    def get_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        x = kl.Dense(self.input_dim, activation=ka.linear)(latent_input)
        decoder = keras.Model(latent_input, x, name="decoder")
        return decoder
