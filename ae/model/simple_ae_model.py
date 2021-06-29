import os
import numpy as np
import pandas as pd
import logging
import h5py
from datetime import datetime
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.activations as ka
from tensorflow.keras import Sequential as ks
from tensorflow.keras import optimizers as ko

# import tensorflow.keras.initializers as ki

from ae.base.base_model import BaseModel


class SimpleAEModel(BaseModel):
    def __init__(self):
        super(SimpleAEModel, self).__init__()
        self.input_dim = None
        self.encoder_input = None
        self.latent_dim = None
        self.hidden_dims = None
        self.units = None
        self.reg1 = None
        self.dropout = None
        self.lr = None
        self.opt = None
        self.loss = None
        self.name = None

        self.encoder = None
        self.decoder = None
        self.model = None





    def init_from_config(self, config):
        self.input_dim = int(config.model.input_dim)
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = int(config.model.latent_dim)
        self.hidden_dims = np.array(config.model.hidden_dims)
        self.units = self.get_units()
        self.reg1 = config.model.reg1
        self.dropout = config.model.dropout
        self.lr = config.model.lr
        self.opt = self.get_opt(config.model.opt)
        self.loss = config.model.loss
        self.name = self.get_name(config.model.name)


    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    def get_name(self, name):
        lr_name = -int(np.log10(self.lr))

        out_name = f'{self.loss}_lr{lr_name}_l{self.latent_dim}_h{len(self.hidden_dims)}'
        if self.dropout != 0:
            out_name = out_name + f'dp{self.dropout}_'
        t = datetime.now().strftime("%m%d_%H%M%S")
        out_name = out_name + name + '_' + t
        return out_name.replace('.', '')

    def get_units(self):
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.latent_dim]
        units = [self.input_dim, *self.hidden_dims, self.latent_dim]
        print(units)
        return units 

    def build_autoencoder(self):
        encoded = self.encoder(self.encoder_input)
        decoded = self.decoder(encoded)
        ae = keras.Model(self.encoder_input, decoded, name="ae")
        self.model = ae

    def build_encoder(self):
        x = kl.Dense(self.hidden_dims[0], activation=ka.tanh, name='en_tanh_0')(self.encoder_input)
        x = kl.Dropout(self.dropout, name='en_dp_0')(x)
        for ii, unit in enumerate(self.hidden_dims[1:]):
            name = 'encod_' + str(ii + 1)
            x = self.add_dense_layer(unit, dp_rate=self.dropout, reg1=self.reg1, name=name)(x)
        x = kl.Dense(self.latent_dim, activation=ka.tanh, name='encode_latent')(x)
        self.encoder = keras.Model(self.encoder_input, x, name="encoder")

    def build_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        x = latent_input
        for ii, unit in enumerate(self.hidden_dims[::-1]):
            name = 'decod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.dropout, reg1=self.reg1, name=name)(x)
        x = kl.Dense(self.input_dim, name='de_last_linear')(x)
        self.decoder = keras.Model(latent_input, x, name="decoder")

    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
        if reg1 is not None and reg1 > 0.0:
            kl1 = kr.l1(reg1)
        else:
            kl1 = None
        layer = ks([kl.Dense(unit, kernel_regularizer=kl1, name=name),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(dp_rate)
                    # keras.activations.tanh()
                    ])
        return layer

    def build_model(self, config):
        self.init_from_config(config)
        self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()

        self.model.compile(
            loss=self.loss,
            optimizer=self.opt,
            metrics=['acc'],
        )


