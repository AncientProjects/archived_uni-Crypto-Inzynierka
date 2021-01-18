class BaseTrain(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, train_data, test_data):
        raise NotImplementedError
