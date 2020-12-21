class DataConfig(object):
    def __init__(self, config):
        self.test_len = config.data_loader.test_len
        self.window_size = config.data_loader.window_size
        self.sequence_len = config.data_loader.sequences

    # split field is just an integer to split all raw values to train and test
    def get_split(self):
        return self.test_len - self.window_size

    def get_window_and_sequence(self):
        return self.window_size, self.sequence_len
