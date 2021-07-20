import os

class BaseModel(object):
    def __init__(self):
        self.model = None
        self.type = None
        self.name = None
        self.save_path = "/home/swei20/AE/trained_model/"

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path=None):
        if self.model is None:
            raise Exception("You have to build the model first.")
        MODEL_PATH=os.path.join(os.environ["home"], "trained_model", self.type, self.name, "")

        # checkpoint_path =f"{self.save_path}/{self.type}/{self.name}/"
        print("Saving model...")
        self.model.save_weights(MODEL_PATH)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError
