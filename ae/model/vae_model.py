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
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
# import tensorflow.keras.initializers as ki
from tensorflow.keras import backend as K


from ae.base.base_model import BaseModel
from .simple_ae_model import SimpleAEModel

class VAEModel(SimpleAEModel):
    def __init__(self):
        super().__init__()
        self.stddev = None # stddev of the latent variable       

    def init_from_config(self, config):
        super().init_from_config(config)
        self.stddev = config.model.stddev or 0.1
        self.name =  "std" + str(self.stddev).replace('.', '') + "_" + self.name

    def build_VAE(self):
        z_mean, z_log_sigma, z = self.encoder(self.encoder_input)
        out = self.decoder(z)
        vae = keras.Model(self.encoder_input, out, name="ae")
        vae_loss = self.get_vae_loss(out, z_mean, z_log_sigma)
        vae.add_loss(vae_loss)
        self.model = vae

    def build_encoder(self):
        h = self.encode(self.encoder_input)
        z_mean = kl.Dense(self.latent_dim)(h)
        z_log_sigma = kl.Dense(self.latent_dim)(h)
        z = kl.Lambda(self.sampling)([z_mean, z_log_sigma])
        self.encoder = keras.Model(self.encoder_input, [z_mean, z_log_sigma, z], name="encoder")
        
    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                mean=0., stddev=self.stddev)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_vae_loss(self, outputs, z_mean, z_log_sigma):
        reconstruction_loss = keras.losses.mse(self.encoder_input, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss 


    def build_model(self, config):
        self.init_from_config(config)
        logging.info(f"NAME: {self.name}")

        self.build_encoder()
        self.build_decoder()
        self.build_VAE()

        self.model.compile(
            # loss=self.loss,
            optimizer=self.opt,
            # metrics=['acc'],
            metrics=[MeanSquaredError()]
            # metrics=[RootMeanSquaredError()]
        )

    