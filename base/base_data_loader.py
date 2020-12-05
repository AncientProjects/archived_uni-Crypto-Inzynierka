class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.window_size = self.config.data_loader.window_size
        self.sequence_len = self.config.data_loader.sequences
        self.test_len = self.config.data_loader.test_len
        self.batch_size = self.config.trainer.batch_size

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

    def get_test_time(self):
        raise NotImplementedError
