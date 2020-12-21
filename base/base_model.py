class BaseModel(object):
    def __init__(self, train_data, config):
        self.train_data = train_data
        self.config = config
        self.window_size = self.config.data_loader.window_size
        self.learning_rate = self.config.model.learning_rate
        self.momentum = self.config.model.momentum
        self.sequence_size = self.config.data_loader.sequences
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load(checkpoint_path)
        print("Model loaded")

    def load_model(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
