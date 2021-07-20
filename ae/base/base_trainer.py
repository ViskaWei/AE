import os
os.environ["home"] = "/home/swei20/AE/"

class BaseTrain(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.root = os.environ["home"]




    def train(self):
        raise NotImplementedError
