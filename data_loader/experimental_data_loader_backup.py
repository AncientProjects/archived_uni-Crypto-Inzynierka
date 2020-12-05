from pandas import read_csv
from base.base_data_loader import BaseDataLoader
from data_loader.data_transformer.data_transformer import DataTransformer
from utils.utils import prepare_supervised_data_only, prepare_data, \
    split_and_reshape


class ExperimentalDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ExperimentalDataLoader, self).__init__(config)
        # self.train_data, self.test_data = load_two_datasets(
        #     "data/bitcoin_1minute_1-1000.csv",
        #     "data/bitcoin_1minute_1001-1201.csv"
        # )
        # self.split_time = self.config.data_loader.split_time
        self.series = read_csv("data/testerino.csv")
        # self.split_time = self.series.shape[0] - 1000

        self.close_raw_all = self.series['close']
        self.time_values = self.series['time']

        # split data to train and test
        # self.split_time = len(self.time_values) - 1000
        # self.train_close_raw = self.close_raw_all[:self.split_time]
        # self.test_close_raw = self.close_raw_all[self.split_time:]
        data_transformer = DataTransformer(self.close_raw_all, self.config)

        # transform train data
        data_transformer.transform_train_data(self.train_raw, self.window_size, self.sequence_len)
        # transform test data

        # transform data (difference -> supervise -> scale)
        self.scaler, self.train_prepared, self.test_prepared = prepare_data(
            self.train_raw,
            self.test_raw,
            self.window_size,
            self.sequence_len)

        # train & test data len is lower by 1 because of the differentiation

        # TOTO rework
        _, self.test = prepare_supervised_data_only(self.close_raw_all, self.window_size, self.sequence_len)
        _, self.test_time = prepare_supervised_data_only(self.time_values, self.window_size, self.sequence_len)

        self.train_scaled_x, self.train_scaled_y = split_and_reshape(self.train_prepared, self.window_size)
        self.test_scaled_x, self.test_scaled_y = split_and_reshape(self.test_prepared, self.window_size)

        self.test_time = self.test_time[:-1, self.window_size:self.window_size+1]
        self.test_x, self.test_y = split_and_reshape(self.test, self.window_size)
        self.test_y = self.test_y[:-1]
        #

    def get_train_data(self):
        return self.train_scaled_x, self.train_scaled_y

    def get_test_data(self):
        return self.test_scaled_x, self.test_scaled_y

    def get_test_values(self):
        return self.test_x, self.test_y

    def get_test_time(self):
        return self.test_time
