import os
import numpy as np
import pandas as pd
import logging
import signal
import h5py
from datetime import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model, save_model
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.initializers as ki
import tensorflow.keras.optimizers as ko
import tensorflow.keras.activations as ka

import ae.util.util as util
from ae.util.constants import Constants
# from pfsspec.data.dataset import Dataset
# from pfsspec.ml.dnn.dnnmodel import DnnModel
#from pfsspec.ml.dnn.keras.radam import RAdam
# from pfsspec.ml.dnn.keras.tensorboard import TensorBoard
# from pfsspec.ml.dnn.keras.constantvector import ConstantVector
# from pfsspec.ml.dnn.keras.basemodelcheckpoint import BaseModelCheckpoint
# from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator
# from pfsspec.ml.dnn.keras.earlystopping import EarlyStopping
# from pfsspec.ml.dnn.keras.keyboardinterrupt import KeyboardInterrupt
# from pfsspec.ml.dnn.keras.periodiccheckpoint import PeriodicCheckpoint

# from .losses import *

class KerasModel(DnnModel):
    def __init__(self, orig=None, levels=4, units=32):
        super(KerasModel, self).__init__(orig=orig)

        self.is_batch = False
        self.threads = 1
        self.name = "unnamed_model"
        self.type = None            # dense, cnn etc.
        self.mode = None            # reg, gen, ae
        self.input_shape = None
        self.output_shape = None
        self.input_slice = None
        self.output_slice = None
        self.include_wave = False
        self.levels = levels
        self.units = units

        self.callbacks = None
        self.history = None

        self.activation = Constants.DNN_DEFAULT_ACTIVATION
        self.loss = Constants.DNN_DEFAULT_LOSS
        self.dropout_rate = Constants.DNN_DEFAULT_DROPOUT_RATE
        self.dropout_input = True
        self.dropout_levels = None
        self.force_dropout = None   # Use dropouts when predicting
        self.kernel_regularization = Constants.DNN_DEFAULT_KERNEL_REGULARIZATION
        self.bias_regularization = Constants.DNN_DEFAULT_BIAS_REGULARIZATION
        self.gpus = None
        self.initial_epoch = 0
        self.epochs = Constants.DNN_DEFAULT_EPOCHS
        self.validation_period = Constants.DNN_DEFAULT_VALIDATION_PERIOD
        self.checkpoint_period = Constants.DNN_DEFAULT_CHECKPOINT_PERIOD
        self.patience = Constants.DNN_DEFAULT_PATIENCE
        self.optimizer = Constants.DNN_DEFAULT_OPTIMIZER
        self.learning_rate_scheduler = None

        # Regularization options
        self.batch_normalization = True

        # Bias initializers
        self.bias = Constants.DNN_DEFAULT_BIAS
        self.bias_initializer = None
        self.output_bias = Constants.DNN_DEFAULT_OUTPUT_BIAS
        self.output_bias_initializer = None

        # MC options
        self.mc_count = None

        self.input_layer = None
        self.output_layer = None
        self.base_model = None
        self.compiled_model = None
        self.history = None

    def get_arg(self, name, old_value, args):
        return util.get_arg(name, old_value, args)

    def is_arg(self, name, args):
        return util.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--name', type=str, help='Model name prefix\n')
        parser.add_argument('--levels', type=int, help='Number of levels\n')
        parser.add_argument('--units', type=int, help='Number of units\n')

        parser.add_argument('--act', type=str, default=Constants.DNN_DEFAULT_ACTIVATION, help='Activation function')
        parser.add_argument('--loss', type=str, default=Constants.DNN_DEFAULT_LOSS, help='Loss function')
        parser.add_argument('--dropout', type=str, default=None, help='Use dropout, no or rate as float.')
        parser.add_argument('--dropout-input', action='store_true', help='Use drop-outs right after input.')
        parser.add_argument('--dropout-levels', type=int, default=None, help='Limit drop-outs to this many levels.')
        parser.add_argument('--reg-kernel', type=float, nargs=2, default=None, help='Kernel regularizer weight L1 L2.')
        parser.add_argument('--reg-bias', type=float, nargs=2, default=None, help='Bias regularizer weight L1 L2.')
        parser.add_argument('--output-bias', type=str, default=None, help='Initialize output bias to constant.')

        parser.add_argument('--gpus', type=str, help='GPUs to use\n')

        parser.add_argument('--epochs', type=int, default=Constants.DNN_DEFAULT_EPOCHS, help='Number of epochs\n')
        parser.add_argument('--validation-period', type=int, default=None, help='Number of epochs between evaluations\n')
        parser.add_argument('--checkpoint-period', type=int, default=None, help='Checkpoint period')
        parser.add_argument('--patience', type=int, default=Constants.DNN_DEFAULT_PATIENCE,
                            help='Number of epochs to wait before early stop.\n')

        parser.add_argument('--opt', type=str, default=Constants.DNN_DEFAULT_OPTIMIZER, help='Optimizer')
        parser.add_argument('--lr-sch', type=str, default=None, help='Learning rate schedule\n')
        parser.add_argument('--lr', type=float, default=None, nargs='+', help='Learning rate parameters\n')
        parser.add_argument('--momentum', type=float, default=None, help='Optimizer momentum\n')
        parser.add_argument('--nesterov', action='store_true', help='Optimizer Nesterov\n')

        parser.add_argument('--no-batchnorm', action='store_true', help='Do not use batch normalization.')

        parser.add_argument('--mc-count', type=int, default=None, help='Monte Carlo dropout realizations.')

    def apply_args(self, args):
        # TODO: delete this?
        #if 'wave' in args and args['wave'] is not None:
        #    self.include_wave = args['wave']
        if 'levels' in args and args['levels'] is not None:
            self.levels = args['levels']
        if 'units' in args and args['units'] is not None:
            self.units = args['units']

        if args['act'] == 'lrelu':
            self.activation = kl.LeakyReLU
        else:
            self.activation = args['act']

        if 'loss' in args:
            if args['loss'] == 'mse_cross':
                self.loss = mean_squared_error_cross
            elif args['loss'] == 'mqe_cross':
                self.loss = mean_quad_error_cross
            else:
                self.loss = args['loss']
        else:
            self.loss = 'mse'

        if 'dropout' in args and args['dropout'] is not None:
            if args['dropout'] == 'no':
                self.dropout_rate = 0
            else:
                self.dropout_rate = float(args['dropout'])
        if 'dropout_input' in args and args['dropout_input'] is not None:
            self.dropout_input = args['dropout_input']
        if 'dropout_levels' in args and args['dropout_levels'] is not None:
            self.dropout_levels = args['dropout_levels']

        if 'reg_kernel' in args and args['reg_kernel'] is not None:
            self.kernel_regularization = args['reg_kernel']
        if 'reg_bias' in args and args['reg_bias'] is not None:
            self.bias_regularization = args['reg_bias']

        if 'output_bias' in args and args['output_bias'] is not None:
            self.output_bias = args['output_bias']

        if 'gpus' in args and args['gpus'] is not None:
            self.gpus = args['gpus']
        if 'epochs' in args and args['epochs'] is not None:
            self.epochs = args['epochs']
        if 'validation_period' in args and args['validation_period'] is not None:
            self.validation_period = args['validation_period']
        if 'checkpoint_period' in args and args['checkpoint_period'] is not None:
            self.checkpoint_period = args['checkpoint_period']
        self.patience = args['patience']

        if 'no_batchnorm' in args and args['no_batchnorm'] is not None:
            self.batch_normalization = not args['no_batchnorm']

        if 'no_batchnorm' in args and args['no_batchnorm'] is not None:
            self.batch_normalization = not args['no_batchnorm']

        self.mc_count = self.get_arg('mc_count', self.mc_count, args)

        self.create_optimizer(args)

    def create_optimizer(self, args):
        lr = None
        decay = None
        momentum = None
        nesterov = None

        with K.name_scope(self.optimizer.__class__.__name__):
            if 'lr_sch' in args and args['lr_sch'] == 'none':
                self.learning_rate_scheduler = None
                if len(args['lr']) > 0:
                    lr = args['lr'][0]
                if len(args['lr']) > 1:
                    decay = args['lr'][1]
            elif 'lr_sch' in args and args['lr_sch'] == 'drop':
                a = 0.1
                b = 0.1
                c = 0.5
                if len(args['lr']) > 0:
                    a = args['lr'][0]
                if len(args['lr']) > 1:
                    b = args['lr'][1]
                if len(args['lr']) > 2:
                    c = args['lr'][2]
                self.learning_rate_scheduler = lambda epoch, lr: a * np.power(c, epoch * b)
            elif 'lr_sch' in args and args['lr_sch'] == 'slowdrop':
                a = 0.1
                b = 0.1
                c = 0.5
                if len(args['lr']) > 0:
                    a = args['lr'][0]
                if len(args['lr']) > 1:
                    b = args['lr'][1]
                if len(args['lr']) > 2:
                    c = args['lr'][2]
                self.learning_rate_scheduler = lambda epoch, lr: a * np.power(c, np.log10(epoch * b + 1))
            elif 'lr_sch' in args and args['lr_sch'] is not None:
                raise Exception('Unknown learning rate schedule')

        if 'momentum' in args and args['momentum'] is not None:
            momentum = args['momentum']
        
        if 'nesterov' in args and args['nesterov'] is not None:
            nesterov = args['nesterov']

        if args['opt'] == 'radam':
            #self.optimizer = RAdam()
            raise NotImplementedError()
        elif args['opt'] == 'adam':
            self.optimizer = ko.adam.Adam(learning_rate=lr or 0.001)
        elif args['opt'] == 'sgd':
            self.optimizer = ko.gradient_descent.SGD(learning_rate=lr or 0.01,
                                    decay=decay or 0.0,
                                    momentum=momentum or 0.0,
                                    nesterov=nesterov or False)   
        #elif type(args['opt']) is str:
        #    self.optimizer = tf.keras.optimizers.get(args['opt'])
        else:
            raise Exception('Unknown optimizer')

    def generate_name(self):
        loss = self.loss

        if type(loss) != str:
            loss = self.loss.__name__

        if loss == 'mean_squared_error':
            loss = 'mse'
        elif loss == 'mean_absolute_error':
            loss = 'mae'
        elif loss == 'mean_absolute_percentage_error':
            loss = 'mape'
        elif loss == 'mean_squared_logarithmic_error':
            loss = 'msle'
        elif loss == 'kullback_leibler_divergence':
            loss = 'kld'
        elif loss == 'cosine_proximity':
            loss = 'cosine'
        elif loss == 'mean_squared_error_cross':
            loss = 'mse_cross'
        elif loss == 'mean_quad_error_cross':
            loss = 'mqe_cross'
        else:
            loss = loss.lower()

        if type(self.optimizer) == str:
            opt = self.optimizer.lower()
        else:
            opt = type(self.optimizer).__name__.lower()

        self.name = '{}_{}_{}_{}_{}'.format(self.type, self.levels, self.units, loss, opt)

    def set_model_shapes(self, data_shape, labels_shape):
        raise NotImplementedError()

    def init_gpus(self, script):
        if not script.debug:
            tf.autograph.set_verbosity(3, False)

        devices = tf.config.list_physical_devices('GPU')
        if not devices:
            raise Exception("No GPU available.")
                    
        tf.compat.v1.disable_eager_execution()

        if self.gpus is not None:
            gpus = self.gpus.split(',')
            devices = [devices[int(i)] for i in gpus]
            tf.config.set_visible_devices(devices, 'GPU')
            self.logger.info('Configured tensorflow to run on devices {}'.format(devices))

        for d in devices:
            tf.config.experimental.set_memory_growth(d, True)
            # config.gpu_options.per_process_gpu_memory_fraction = args.reserve_vram

    def release_gpus(self, script):
        tf.keras.backend.clear_session()

    def create_model(self):
        # TODO: Why do we reset bias initializer here?
        self.bias_initializer = None

        if self.bias is not None:
            self.bias_initializer = ki.Constant(self.bias)

        return None

    def set_output_bias(self, training_generator=None):
        if training_generator is not None and training_generator.output_scale is not None:
            self.output_bias = 0.0
        elif self.output_bias is not None:
            self.output_bias = float(self.output_bias)

        if self.output_bias is not None:
            if isinstance(self.output_bias, np.ndarray):
                self.output_bias_initializer = ConstantVector(self.output_bias)
            elif type(self.output_bias) is float:
                self.output_bias_initializer = ki.Constant(self.output_bias)
            else:
                self.output_bias_initializer = ki.random_uniform()

        self.logger.info('Initializing output bias to {}'.format(self.output_bias))

    def compile_model(self):
        if self.gpus is not None and len(self.gpus.split(',')) > 1:  # multi gpu model
            self.logger.info('Compiling model for multiple gpus.')
            self.compiled_model = multi_gpu_model(self.base_model, gpus=len(self.gpus.split(',')))
        else:
            self.logger.info('Compiling model for single gpu.')
            self.compiled_model = self.base_model
        self.compiled_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.loss])

        self.logger.info('Loss: {}, optimizer: {}'.format(self.loss, self.optimizer))
        self.logger.info('Optimizer is {} with parameters {}'.format(type(self.compiled_model.optimizer).__name__, self.compiled_model.optimizer.get_config()))

    def get_kernel_regularizer(self):
        if self.kernel_regularization is not None:
            return kr.L1L2(*self.kernel_regularization)
        else:
            return None

    def get_bias_regularizer(self):
        if self.bias_regularization is not None:
            return kr.L1L2(*self.bias_regularization)
        else:
            return None

    def add_input_layer(self):
        x = self.input_layer = kl.Input(self.input_shape)
        return x

    def add_dropout_layer(self, il, x, dropout_rate=None):
        dropout_rate = dropout_rate or self.dropout_rate
        if dropout_rate > 0 and \
            ((il == 0 and self.dropout_input) or (il > 0 and (self.dropout_levels is None or il <= self.dropout_levels))):
            x = kl.Dropout(rate=dropout_rate)(x, training=self.force_dropout)
        return x

    def add_dense_layer(self, il, x, units=None, bias_initializer=None):
        units = units or self.units
        bias_initializer = bias_initializer or self.bias_initializer
        x = kl.Dense(units,
                     kernel_regularizer=self.get_kernel_regularizer(),
                     bias_regularizer=self.get_bias_regularizer(),
                     use_bias=bias_initializer is not None,
                     bias_initializer=bias_initializer)(x)
        # TODO: add regularization                     
        return x

    def add_reshape_layer(self, x, shape):
        x = kl.Reshape(shape)(x)
        return x

    def add_activation_layer(self, il, x, activation=None):
        activation = activation or self.activation
        if type(activation) is str:
            x = kl.Activation(activation)(x)
        elif type(activation) is type:
            x = activation()(x)
        else:
            raise NotImplementedError()
        return x

    def add_batch_normalization_layer(self, il, x):
        if self.batch_normalization:
            x = kl.BatchNormalization()(x)
        return x

    def add_flatten_layer(self, x):
        x = kl.Flatten()(x)
        return x

    def add_input_layer(self):
        self.input_layer = x = kl.Input(self.input_shape)
        return x

    def add_output_layer(self, x):
        if self.output_bias_initializer is not None:
            self.output_layer = x = kl.Dense(self.output_shape,
                                         use_bias=True,
                                         bias_initializer=self.output_bias_initializer)(x)
        else:
            self.output_layer = x = kl.Dense(self.output_shape, use_bias=True)(x)
        return x

    def print(self):
        print((self.base_model or self.compiled_model).summary())

    def freeze(self, model=None):
        # Freeze model weights in every layer.
        # see: https://stackoverflow.com/questions/51944836/keras-load-model-valueerror-axes-dont-match-array
        if model is None:
            model = self.base_model
        for layer in model.layers:
            layer.trainable = False
            if isinstance(layer, models.Model):
                self.freeze(layer)

    def save_json(self, filename):
        model_json = (self.base_model or self.compiled_model).to_json()
        with open(filename, "w") as json_file:
            json_file.write(model_json)

    def load_json(self, filename):
        with open(filename, "r") as json_file:
            model = json_file.read()
        self.base_model = model_from_json(model)
        self.compile_model()

    def save_state(self, filename):
        self.logger.info('Saving model to {}'.format(filename))
        save_model(self.compiled_model, filename)

        config = self.compiled_model.optimizer.get_config()
        with h5py.File(filename, 'a') as f:
            grp = f['optimizer_weights'].create_group('config')
            for k in config:
                grp.create_dataset(k, data=config[k])

    def load_state(self, filename):
        self.logger.info('Loading model from {}'.format(filename))
        self.compiled_model = load_model(filename)

        self.logger.info('Loaded model from {}'.format(filename))
        self.logger.info('Optimizer is {} with parameters {}'.format(type(self.compiled_model.optimizer).__name__, self.compiled_model.optimizer.get_config()))

    def load_state_optimizer(self, filename):
        # This is obsolete as TF1.15.0 seems to handle it correctly
        with h5py.File(filename, 'a') as f:
            if 'optimizer_weights' in f and 'config' in f['optimizer_weights']:
                grp = f['optimizer_weights']['config']
                for k in grp.keys():
                    if hasattr(self.compiled_model.optimizer, 'k'):
                        o = getattr(self.compiled_model.optimizer, 'k')
                        if isinstance(o, K.variable):
                            K.set_value(o, config[k][()])
                        else:
                            setattr(self.compiled_model.optimizer, 'k', config[k][()])

    def load_weights(self, filename=None, mode='test'):
        filename = self.get_weights_path(filename, mode)
        model = self.base_model or self.compiled_model
        self.logger.info('Loading model weights from {}'.format(filename))
        model.load_weights(filename)

    def save_history(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'a') as f:
                self.history.to_csv(f, index=False, header=False)
        else:
            with open(filename, 'w') as f:
                self.history.to_csv(f, index=False)

    def load_history(self, filename):
        if os.path.isfile(filename):
            self.logger.info('Loading history from {}'.format(filename))
            with open(filename, 'r') as f:
                self.history = pd.read_csv(f)

            self.initial_epoch = self.history.shape[0]
            self.callbacks['checkpoint_train'].best = self.history.iloc[-1]['loss']
            self.callbacks['checkpoint_test'].best = self.history.iloc[-1]['val_loss']
        else:
            self.logger.info('Cannot find history file {}.'.format(filename))

    def ensure_model_created(self):
        if self.base_model is None:
            self.logger.info('Creating model with shapes: {} {}'.format(self.input_shape, self.output_shape))
            self.create_model()
        if self.input_slice is not None:
            self.logger.info('Input must be sliced down to {}'.format(self.input_slice))
        if self.output_slice is not None:
            self.logger.info('Output must be sliced down to {}'.format(self.output_slice))
        if self.compiled_model is None:
            self.compile_model()

    def get_tensorboard_path(self):
        #return os.path.join(os.path.dirname(self.checkpoint_path), 'tb')
        return self.checkpoint_path

    def get_weights_path(self, filename=None, mode='test'):
        if filename is not None:
            filename = os.path.join(self.checkpoint_path, filename)
        elif mode is not None:
            if mode == 'train':
                filename = 'best_train_weights.h5'
            elif mode == 'test':
                filename = 'best_test_weights.h5'
            elif mode == 'epoch':
                filename = 'weights_{}.h5'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return os.path.join(self.checkpoint_path, filename)

    def create_callbacks(self, script):
        self.callbacks = {}

        # Slurm kills job if it's unresponsive to SIGINT, so use SIGURS1 for
        # graceful termination of the learning process
        s = signal.SIGUSR1 if self.is_batch else signal.SIGINT
        self.callbacks['interrupt'] = KeyboardInterrupt(signals=[s,])

        if self.checkpoint_path is not None:
            tbpath = self.get_tensorboard_path()
            self.callbacks['tensorboard'] = TensorBoard(log_dir=tbpath)

            trainpath = self.get_weights_path(mode='train')
            testpath = self.get_weights_path(mode='test')

            # Save best model weights after each evaluation period
            #if self.gpus is not None and len(self.gpus.split(',')) > 1:
            #    train_cp = BaseModelCheckpoint(trainpath, period=self.validation_period, monitor='loss', save_best_only=True, verbose=1, base_model=self.base_model)
            #    test_cp = BaseModelCheckpoint(testpath, period=self.validation_period, monitor='val_loss', save_best_only=True, verbose=1, base_model=self.base_model)
            #else:

            train_cp = ModelCheckpoint(trainpath, monitor='loss', save_best_only=True, verbose=1)
            test_cp = ModelCheckpoint(testpath, monitor='val_loss', save_best_only=True, verbose=1)

            self.callbacks['checkpoint_train'] = train_cp
            self.callbacks['checkpoint_test'] = test_cp

        # Call at fixed intervals, do whatever script checkpoint loopback does
        if self.checkpoint_period is not None and script is not None and self.checkpoint_path is not None:
            self.callbacks['checkpoint_periodic'] = PeriodicCheckpoint(self.checkpoint_period, lambda epoch: script.checkpoint(epoch))

        # Stop training after a number of epochs or when NaNs are detected
        self.callbacks['early_stopping'] = EarlyStopping(patience=self.patience, verbose=1)

        if self.learning_rate_scheduler is not None:
            self.callbacks['learning_rate_scheduler'] = LearningRateScheduler(self.learning_rate_scheduler, verbose=1)

    def train(self, training, validation):
        self.train_with_generator(training, validation)

    def train_with_generator(self, data_generator, validation_generator):
        if not data_generator.augmenter.shuffle:
            self.logger.warning('Shuffling is not enabled on data generator {}'.format(data_generator.augmenter.mode))
        
        data_generator.augmenter.input_reshape = self.input_shape
        validation_generator.augmenter.input_reshape = self.input_shape

        if type(self.callbacks) is dict:
            callbacks = [self.callbacks[k] for k in self.callbacks]

        # TODO: control from arguments?
        if self.is_batch:
            verbose = 2     # one per epoch
        else:
            verbose = 1     # progress bar

        if self.initial_epoch == 0:
            self.logger.info("Starting training process...")
        else:
            self.logger.info("Continuing training process at epoch {}...".format(self.initial_epoch))

        results = self.compiled_model.fit(x=data_generator,
                                          steps_per_epoch=data_generator.batch_count,
                                          validation_data=validation_generator,
                                          validation_steps=validation_generator.batch_count,
                                          validation_freq=self.validation_period,
                                          epochs=self.epochs,
                                          initial_epoch=self.initial_epoch,
                                          verbose=verbose,
                                          callbacks=callbacks,

                                          # Never use multiprocessing here, it will be
                                          # done by the custom data generator
                                          workers=0,
                                          use_multiprocessing=False,

                                          # Shuffling is done by the data generator, do not
                                          # request random order of batches here because that
                                          # would break the cache logic used for large datasets
                                          shuffle=False)        
                
        # TODO: if validation_freq is >1, history needs processing before converting to
        #       a DataFrame because entiries in the lists don't have the same number
        
        self.history = pd.DataFrame(results.history)

    def predict(self, input):
        # Input and outputs arrays must be scaled manually!
        # self.logger.info('Predicting from array...')

        if self.mc_count is None:
            prediction = self.compiled_model.predict(input)
            return prediction
        else:
            res = np.empty((input.shape[0], self.output_shape) + (self.mc_count,))
            for i in range(self.mc_count):
                prediction = self.compiled_model.predict(input)
                res[:, :, i] = prediction
            return res
            
    def append_dense_layers(self, input_layer, levels, units):
        x = input_layer
        while levels > 0:
            if self.dropout_rate > 0:
                x = kl.Dropout(rate=self.dropout_rate)(x)
            x = kl.Dense(units)(x)
            if self.batch_normalization:
                x = kl.BatchNormalization()(x)
            x = kl.Activation(self.activation)(x)
            levels -= 1
        return x

    def append_dense_pyramid(self, input_layer, levels, units):
        x = input_layer
        while levels > 0:
            if self.dropout_rate > 0:
                x = kl.Dropout(rate=self.dropout_rate)(x)
            x = kl.Dense(units)(x)
            if self.batch_normalization:
                x = kl.BatchNormalization()(x)
            x = kl.Activation(self.activation)(x)
            levels -= 1
            units //= 2
        return x