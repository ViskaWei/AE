import os
import h5py
# TODO: which portion should be in base?
from ae.base.base_data_loader import BaseDataLoader

class SpecDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpecDataLoader, self).__init__(config)
        fn = os.path.join(config.data.dir, config.data.filename)
        with h5py.File(fn, 'r') as f:
            train_data = f[config.data.train][()]
            test_data = f[config.data.test][()]

        self.X_train = train_data
        self.X_test = test_data


    def get_train_data(self):
        return self.X_train, self.X_train

    def get_test_data(self):
        return self.X_test, self.X_test

