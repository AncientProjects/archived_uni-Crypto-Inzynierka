class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_data(self, i):
        raise NotImplementedError

    def get_test_data(self, i):
        raise NotImplementedError

    def get_test_time_values(self, i):
        raise NotImplementedError
