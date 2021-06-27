import numpy as np

class Constants():
    DNN_DEFAULT_ACTIVATION = 'relu'
    DNN_DEFAULT_KERNEL_REGULARIZATION = [0, 5e-5]
    DNN_DEFAULT_BIAS_REGULARIZATION = [0, 5e-5]
    DNN_DEFAULT_LOSS = 'mean_squared_error'
    DNN_DEFAULT_VALIDATION_SPLIT = 0.2
    DNN_DEFAULT_EPOCHS = 100
    DNN_DEFAULT_CHECKPOINT_PERIOD = 100
    DNN_DEFAULT_VALIDATION_PERIOD = 1
    DNN_DEFAULT_PATIENCE = 1000
    DNN_DEFAULT_BATCH_SIZE = 16
    DNN_DEFAULT_OPTIMIZER = 'adam'
    DNN_DEFAULT_DROPOUT_RATE = 0.02
    DNN_DEFAULT_DECAY = 0
    DNN_DEFAULT_BIAS = 0.1
    DNN_DEFAULT_OUTPUT_BIAS = 0.5