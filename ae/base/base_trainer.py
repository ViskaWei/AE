class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.root = "/home/swei20/AE/"




    def train(self):
        raise NotImplementedError
