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
        super(VAEModel, self).__init__()
        self.model_name = 'vae_model'

    def init_from_config(self, config):
        return super().init_from_config(config)

    def create_model(self, input_shape, output_shape, **kwargs):
        self.model = ks.Sequential()    
        self.model.add(kl.InputLayer(input_shape=input_shape))
        self.model.add(kl.Flatten())
        self.model.add(kl.Dense(units=128, activation=ka.relu, kernel_regularizer=kr.l2(0.01)))
        self.model.add(kl.Dense(units=64, activation=ka.relu, kernel_regularizer=kr.l2(0.01)))

    def build_encoder(self):
        x = self.encode(self.encoder_input)
        z_mean = kl.Dense(latent_dim)(h)
        z_log_sigma = kl.Dense(latent_dim)(h)
        self.encoder = 
        
    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon