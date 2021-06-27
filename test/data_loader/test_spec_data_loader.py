import numpy as np
from test.test_base import TestBase

# import logging

from ae.data_loader.spec_data_loader import SpecDataLoader

from ae.util.args import get_args
from ae.util.config import process_config

class TestSpecDataLoader(TestBase):

    def test_get_train_data(self):
        CONFIG_PATH = '/home/swei20/AE/configs/ae/train/config.json'
        args = get_args(default=CONFIG_PATH)
        config = process_config(args.config)
        data_loader = SpecDataLoader(config)
        data_loader.get_train_data()
        data_on_file = np.array([[1,2,3],[4,5,6]])
        assert len(data_loader.X_train) == 2
        assert data_loader.X_train[0] == data_on_file

        # data_loader.get_test_data()
        # assert len(data_loader.X_test) == 2
        # assert data_loader.X_test[0] == data_on_file

# if __name__ == '__main__':
#     unittest.main()