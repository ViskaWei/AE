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
    def __init__(self, input_dim=4096, latent_dim=32, hidden_dims=[], reg1=None,\
                         encoder_dp=0., loss='mae', lr=0.01, opt='adam', name=''):
        # super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        self.latent_dim = int(latent_dim)
        self.hidden_dims = np.array(hidden_dims)
        self.units = self.get_units()

        self.reg1 = reg1
        self.encoder_dp=encoder_dp

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.lr = lr
        self.opt = self.get_opt(opt)
        self.loss = loss
        self.name = self.get_name(name)
        self.log_dir = "logs/fit/" + self.name

        self.ae = self.get_ae()

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
        if self.encoder_dp != 0:
            out_name = out_name + f'dp{self.encoder_dp}_'
        t = datetime.datetime.now().strftime("%m%d-%H%M%S")
        out_name = out_name + name + '_' + t
        return out_name.replace('.', '')


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
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.latent_dim]
        units = [self.input_dim, *self.hidden_dims, self.latent_dim]
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
        x = self.encoder_input
        for ii, unit in enumerate(self.units[1:]):
            name = 'encod' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.encoder_dp, reg1=self.reg1, name=name)(x)
        encoder = keras.Model(self.encoder_input, x, name="encoder")
        return encoder

    def get_decoder(self):
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

