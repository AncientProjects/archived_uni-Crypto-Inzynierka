class BaseTrain(object):
    def __init__(self, model, train_data, config):
        self.model = model
        self.train_data = train_data
        self.config = config

    def train(self):
        raise NotImplementedError
