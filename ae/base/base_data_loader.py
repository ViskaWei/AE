import numpy as np

class BaseDataLoader(object):
    def __init__(self):
        self.x_train = None
        self.x_test = None
        pass

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError


    def get_centered_data(self, x):
        x_mean =np.mean(x, axis = 0)
        assert x_mean.shape == x[0].shape
        x_scaled = x -  x_mean
        return x_scaled
        
    # def get_std_scaled_data(self, r, x):
    #     x_std = x.std(0)
    #     assert x_std.shape == x[0].shape
    #     std_scaled = x_std ** r
    #     x_std_scaled = x / std_scaled
    #     return x_std_scaled

    # def get_max_norm(self, x):
    #     vmax = np.max(abs(x))
    #     norm_x = x / vmax
    #     assert((np.min(norm_x) >= -1.0) & (np.max(norm_x) <= 1.0))
    #     return norm_x

    # def get_max_norm_data(self, x, bound_in_01=False):
    #     x = self.get_max_norm(x)
    #     if bound_in_01:
    #         x = x / 2.0 + 0.5
    #     return x

    # def get_mean_norm_data(self, x):
    #     mean = np.median(x, axis=0)
    #     std = np.std(x, axis=0)
    #     x  = (x - mean) / std
    #     return x