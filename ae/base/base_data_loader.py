class BaseDataLoader(object):
    def __init__(self):
        pass

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError
