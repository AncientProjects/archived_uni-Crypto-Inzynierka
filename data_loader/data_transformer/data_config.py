class DataConfig(object):
    def __init__(self, config, series_len):
        self.series_len = series_len
        self.window_size = config.data_loader.window_size
        self.sequence_len = config.data_loader.sequences
        self.k = config.data_loader.k_fold
        self.recursive = config.data_loader.recursive
        if self.recursive:
            self.recursive_seq = config.data_loader.recursive_seq
