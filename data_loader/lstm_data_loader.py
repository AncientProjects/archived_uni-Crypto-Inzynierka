from pandas import read_csv
from base.base_data_loader import BaseDataLoader
from utils.utils import prepare_supervised_data_only, prepare_data, \
    split_and_reshape


class LSTMDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(LSTMDataLoader, self).__init__(config)
        # self.train_data, self.test_data = load_two_datasets(
        #     "data/bitcoin_1minute_1-1000.csv",
        #     "data/bitcoin_1minute_1001-1201.csv"
        # )
        self.split_time = self.config.data_loader.split_time
        self.series = read_csv("data/bitcoin_1minute_10001-20000.csv")

        self.close_raw_values = self.series['Close'].values
        self.time_values = self.series['Unix Timestamp'].values

        self.scaler, self.train_scaled, self.test_scaled, self.num_of_tensors = prepare_data(
            self.close_raw_values,
            self.window_size,
            self.sequence_len,
            self.split_time)

        _, self.test = prepare_supervised_data_only(self.close_raw_values, self.window_size, self.sequence_len,
                                                    self.num_of_tensors)
        _, self.test_time = prepare_supervised_data_only(self.time_values, self.window_size,
                                                         self.sequence_len,
                                                         self.num_of_tensors)

        self.train_scaled_x, self.train_scaled_y = split_and_reshape(self.train_scaled, self.window_size)
        self.test_scaled_x, self.test_scaled_y = split_and_reshape(self.test_scaled, self.window_size)

        self.test_time = self.test_time[:-1, self.window_size:self.window_size+1]
        self.test_x, self.test_y = split_and_reshape(self.test, self.window_size)
        self.test_y = self.test_y[:-1]

    def get_train_data(self):
        return self.train_scaled_x, self.train_scaled_y

    def get_test_data(self):
        return self.test_scaled_x, self.test_scaled_y

    def get_test_values(self):
        return self.test_x, self.test_y

    def get_test_time(self):
        return self.test_time
