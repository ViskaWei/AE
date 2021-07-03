import os
import h5py
import numpy as np
from ae.base.base_data_loader import BaseDataLoader

class PcDataLoader(BaseDataLoader):
    def __init__(self):
        super(PcDataLoader, self).__init__()


    def init_from_config(self, config):
        fn = os.path.join(config.data.dir, config.data.filename)
        with h5py.File(fn, 'r') as f:
            train_data = f[config.data.train][()]
        
        # if config.data.std_rate != 0.0:
        #     train_data = self.get_std_scaled_data(config.data.std_rate, train_data)

        if "test" in config.data:
            with h5py.File(fn, 'r') as f:
                test_data = f[config.data.test][()]
        else:
            test_data = None 


        self.x_train = train_data
        self.x_test = test_data



    def get_train_data(self):
        return self.x_train, self.x_train

    def get_test_data(self):
        return self.x_test, self.x_test

