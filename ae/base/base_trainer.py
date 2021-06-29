class BaseTrain(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.root = "/home/swei20/AE/"




    def train(self):
        raise NotImplementedError
