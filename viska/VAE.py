import os
import numpy as np
import pandas as pd
import logging
# import signal
import h5py
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
# from tensorflow.keras import Model
import tensorflow.keras.layers as kl
from tensorflow.keras import Sequential as ks

# import tensorflow.keras.regularizers as kr
# import tensorflow.keras.initializers as ki
from tensorflow.keras import optimizers as ko
# import tensorflow.python.keras.optimizer_v2 as ko
import tensorflow.keras.activations as ka


# class DenseVAE(object):


class model():
    def __init__(self, latent_dim=16, lr=0.001):
        self.input_dim = 4096
        self.latent_dim = latent_dim
        self.lr = lr
        self.hidden_units = [512, 32]

        self.units = self.get_units()

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.vae = None

        self.run()

    def run(self):
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=ko.Adam(learning_rate=self.lr))
    # def set_model_shapes(self, input_shape, latent_size):
    #     self.input_shape = (input_shape[1], )
    #     self.latent_size = latent_size
    def get_units(self):
        # return [self.input_dim, self.input_dim//2, self.latent_dim]
        units = [512, 128]
        return [self.input_dim, *units, self.latent_dim]


    def get_encoder(self):
        encoder_input = keras.Input(shape = (self.input_dim, ), name='spec')
        # x = kl.Flatten()(encoder_input) #dense layer\
        x = encoder_input
        # x = kl.Dense(self.latent_dim, activation="relu")(x)
        # x = kl.BatchNormalization()(x)
        # x = kl.LeakyReLU()(x)
        for ii, unit in enumerate(self.units[1:]):
            x = self.add_dense_layer(unit)(x)
        #     # if ii >= 1:
            #     x = kl.Dropout(0.05, seed=922)
        z_mean = kl.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = kl.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()
        return encoder

    def get_decoder(self):
        latent_input = keras.Input(shape= (self.latent_dim,))
        x = latent_input
        # x = kl.Dense(self.input_dim, activation="relu")(x)
        for ii, unit in enumerate(self.units[-2::-1]):
            x = self.add_dense_layer(unit)(x)
        decoder = keras.Model(latent_input, x, name="decoder")
        # decoder.summary()
        return decoder

    def add_dense_layer(self, unit):
        layer = ks([kl.Dense(unit),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(0.2)
                    ])
        return layer

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_fn = keras.losses.mse
        # self.reconstruction_loss_fn = keras.losses.binary_crossentropy

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # print(data.shape, reconstruction.shape)
            summ = self.reconstruction_loss_fn(data, reconstruction)
            # reconstruction_loss = tf.reduce_mean(summ)
            reconstruction_loss = tf.reduce_sum(summ)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class Sampling(kl.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon