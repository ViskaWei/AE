import os
import numpy as np
import pandas as pd
import logging
import h5py
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras import Sequential as ks
import tensorflow.keras.regularizers as kr
# import tensorflow.keras.initializers as ki
from tensorflow.keras import optimizers as ko
import tensorflow.keras.activations as ka
from ae.base.base_model import BaseModel


class SimpleAEModel(BaseModel):
    def __init__(self, config):
        super(SimpleAEModel, self).__init__(config)
        self.build_model()
        self.input_dim = int(config.model.input_dim)
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = int(config.model.latent_dim)
        self.hidden_dims = np.array(config.model.hidden_dims)
        self.units = self.get_units()

        self.reg1 = reg1
        self.dropout = config.model.dropout

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.lr = config.model.lr
        self.opt = self.get_opt(config.model.opt)
        self.loss = config.model.loss
        self.name = self.get_name(config.model.name)
        self.log_dir = "logs/fit/" + self.name

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6),
            tf.keras.callbacks.ReduceLROnPlateau('loss', 
                                                patience=3, 
                                                min_lr=0., 
                                                factor=0.1),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        ]


    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    def get_name(self, name):
        lr_name = -np.log10(self.lr)

        out_name = f'{self.loss}_lr{lr_name}_l{self.latent_dim}_h{len(self.hidden_dims)}'
        if self.dropout != 0:
            out_name = out_name + f'dp{self.dropout}_'
        t = datetime.now().strftime("%m%d-%H%M%S")
        out_name = out_name + name + '_' + t
        return out_name.replace('.', '')

    def get_units(self):
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.latent_dim]
        units = [self.input_dim, *self.hidden_dims, self.latent_dim]
        print(units)
        return units 

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_encoder(self):
        x = self.encoder_input
        for ii, unit in enumerate(self.units[1:]):
            name = 'encod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1, name=name)(x)
        encoder = keras.Model(self.encoder_input, x, name="encoder")
        return encoder

    def build_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        x = latent_input
        for ii, unit in enumerate(self.hidden_dims[::-1]):
            name = 'decod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1, name=name)(x)
        x = kl.Dense(self.input_dim, name='last')(x)
        decoder = keras.Model(latent_input, x, name="decoder")
        return decoder

    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
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

    def build_model(self):
        self.model = keras.Model(self.encoder_input, self.call(self.encoder_input), name="ae")
        self.model.compile(
            loss=self.loss,
            optimizer=self.opt,
            metrics=['acc'],
        )


