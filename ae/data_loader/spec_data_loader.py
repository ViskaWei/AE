import os
import h5py
import numpy as np
from ae.base.base_data_loader import BaseDataLoader

class SpecDataLoader(BaseDataLoader):
    def __init__(self):
        super(SpecDataLoader, self).__init__()
        self.X_train = None
        self.X_test = None

    def init_from_config(self, config):
        fn = os.path.join(config.data.dir, config.data.filename)
        with h5py.File(fn, 'r') as f:
            train_data = f[config.data.train][()]
        if config.data.norm == "no":
            self.X_train = train_data
        elif config.data.norm == "max":
            self.X_train = self.get_max_norm_data(train_data)
        elif config.data.norm == "max01":
            self.X_train = self.get_max_norm_data(train_data, bound_in_01=True)
        elif config.data.norm == "mean":
            self.X_train = self.get_mean_norm_data(train_data)
        else:
            raise NotImplementedError

        if "test" in config.data:
            with h5py.File(fn, 'r') as f:
                test_data = f[config.data.test][()]
            if config.data.norm == "max":
                self.X_test = self.get_norm_data(test_data)
            elif config.data.norm == "no":
                self.X_test = test_data
            else:
                raise NotImplementedError

    def get_train_data(self):
        return self.X_train, self.X_train

    def get_test_data(self):
        return self.X_test, self.X_test


    def get_max_norm(self, x):
        vmax = np.max(abs(x))
        norm_x = x / vmax
        assert((np.min(norm_x) >= -1.0) & (np.max(norm_x) <= 1.0))
        return norm_x

    def get_max_norm_data(self, x, bound_in_01=False):
        x = self.get_max_norm(x)
        if bound_in_01:
            x = x / 2.0 + 0.5
        return x

    def get_mean_norm_data(self, x):
        mean = np.median(x, axis=0)
        std = np.std(x, axis=0)
        x  = (x - mean) / std
        return x