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
        self.bn = config.model.batchnorm
        self.act_in = config.model.act_in
        self.act_em = config.model.act_em
        self.act_hd = config.model.act_hd
        self.aug = config.model.aug

        self.name = self.get_name(config.model.name)
        logging.info(f"NAME: {self.name}")

    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    def get_name(self, name):
        lr_name = -int(np.log10(self.lr))
        out_name = f'{self.loss}_lr{lr_name}_l{self.latent_dim}_'
        for hid_dim in self.hidden_dims:
            out_name = out_name + "h" + str(hid_dim) + "_"
        t = datetime.now().strftime("%m%d_%H%M%S")
        
        if self.aug:
            dp_name = "" if self.dropout == 0 else f'_dp{self.dropout}_'
            act_name = f"IN{self.act_in[:2]}EM{self.act_em[:2]}HD{self.act_hd[:2]}"
            out_name = out_name + dp_name  + act_name + '_' + name + t
        else:
            out_name = out_name + name + t
        return out_name.replace('.', '')

    def get_units(self):
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.latent_dim]
        units = [self.input_dim, *self.hidden_dims, self.latent_dim]
        logging.info(f"Layers: {units}")
        return units 

    def build_autoencoder(self):
        encoded = self.encoder(self.encoder_input)
        decoded = self.decoder(encoded)
        ae = keras.Model(self.encoder_input, decoded, name="ae")
        self.model = ae

    def build_encoder(self):
        x = self.encoder_input
        if len(self.hidden_dims) > 0:
            x = kl.Dense(self.units[1], kernel_regularizer=kr.l2(self.reg1), name='encode_in')(x)
            
            if self.aug: 
                x = self.add_activation_layer(self.act_in)(x)
                # x = kl.Dropout(self.dropout)(x)
                if self.bn: 
                    x = kl.BatchNormalization()(x)
            for ii, unit in enumerate(self.hidden_dims[1:]):
                name = 'encod_u' + str(unit)
                x = kl.Dense(unit, kernel_regularizer=kr.l2(self.reg1), name=name)(x)
                if self.aug: 
                    x = self.add_activation_layer(self.act_hd)(x)
                    if self.bn: 
                        x = kl.BatchNormalization()(x)
                #   x = kl.Dropout(self.dropout)(x)

        x = kl.Dense(self.latent_dim, kernel_regularizer=kr.l2(self.reg1), name='embed_in')(x)
        if self.aug: 
            x = self.add_activation_layer(self.act_em)(x)
            if self.bn: 
                x = kl.BatchNormalization()(x)

        self.encoder = keras.Model(self.encoder_input, x, name="encoder")

    def build_decoder(self):
        latent_input = keras.Input(shape=(self.latent_dim,))
        x = latent_input
        if len(self.hidden_dims) > 0:        
            x = kl.Dense(self.hidden_dims[-1], kernel_regularizer=kr.l2(self.reg1), name='embed_out')(x)
                
            if self.aug: 
                x = self.add_activation_layer(self.act_hd)(x)
                if self.bn: 
                    x = kl.BatchNormalization()(x)
            for ii, unit in enumerate(self.hidden_dims[::-1][1:]):
                name = 'decod_u' + str(unit)
                x = kl.Dense(unit, kernel_regularizer=kr.l2(self.reg1), name=name)(x)
                
                if self.aug: 
                    x = self.add_activation_layer(self.act_hd)(x)
                    # x = kl.Dropout(self.dropout)(x)
                    if self.bn: 
                        x = kl.BatchNormalization()(x)
        x = kl.Dense(self.input_dim, kernel_regularizer=kr.l2(self.reg1), name='decod_out')(x)
        self.decoder = keras.Model(latent_input, x, name="decoder")


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


    def add_activation_layer(self, act):
        if act == "leaky":
            layer =  kl.LeakyReLU()
        else:
            try:
                layer =  kl.Activation(act)
            except:
                raise NotImplementedError
        return layer




    # def add_dense_layer(self, x, unit, dp_rate=0., reg1=None, name=None):
    #     if reg1 is not None and reg1 > 0.0:
    #         kl1 = kr.l1(reg1)
    #     else:
    #         kl1 = None
        
    #         x = kl.Dense(unit, kernel_regularizer=kl1, name=name)(x)
    #         # kl.Dense(unit, activation=self.act_hd, kernel_regularizer=kl1, name=name),
    #         # kl.BatchNormalization(),
    #         x = kl.LeakyReLU()(x),
    #         if dp_rate > 0.0:
    #             x = kl.Dropout(dp_rate)(x)
    #         # keras.activations.tanh()
    #     return x


        # def get_activation(self, act):
    #     if act == "tanh":
    #         act_fn =  ka.tanh()
    #     elif act == "linear":
    #         act_fn = ka.linear
    #     elif act == "sig":
    #         act_fn = ka.sigmoid
    #     elif act == "relu":
    #         act_fn = ka.relu
    #     return act_fn