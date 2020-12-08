import numpy as np
from pandas import Series, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

from data_loader.data_transformer.split_to_train_and_test import SplitToTrainAndTest
from data_loader.data_transformer.prepare_data import PrepareData


class DataTransformer(object):
    def __init__(self, series_df, config):
        self.config = config
        self.prepare_data = PrepareData(config, series_df)
        # self.close_raw_all = series_df['close']
        # self.time_values = series_df['time']
        # raw, real values straight from data
        self.raw_series = self.close_raw_all
        self.test_len, self.window_len, self.sequence_len = self.load_from_config()
        # split field is just an integer to split all raw values to train and test
        self.split = self.test_len - self.window_len
        self.test_time = self.time_values[-self.test_len + self.window_len:]

        # raw_train and raw_test Series with real values from data before any transformations
        self.raw_split = SplitToTrainAndTest(self.raw_series, self.split)

        # difference -> supervised -> fit scaler to train values and then transform test with it
        # save the scaler for test values
        self.scaler, self.train_transformed, self.test_transformed = self.transform_train_and_test_data()
        self.test_x, self.test_y = self.split_and_reshape(self.test_transformed)

    def load_from_config(self):
        return self.config.data_loader.test_len, \
               self.config.data_loader.window_size, \
               self.config.data_loader.sequences

    def transform_train_and_test_data(self):
        diff_series = self.diff_and_supervised(self.raw_series)
        supervised_split = SplitToTrainAndTest(diff_series, self.split)
        scaler, train_transformed = self.transform_train_data(supervised_split.train)
        return scaler, train_transformed, self.transform_test_data(supervised_split.test, scaler)

    def transform_train_data(self, supervised_values):
        # Series of tuples (index, value) to ndarray of values

        # scaler fit on supervised train ndarray of shape = (num_of_vectors, window_len + sequence_len)
        scaler, train_transformed = self.scale_train(supervised_values)
        return scaler, train_transformed

    def transform_test_data(self, supervised_values, scaler):
        # Series of tuples (index, value) to ndarray of values

        # transform differenced and supervised test values using scaler fit to supervised train values
        test_transformed = self.scale_test(supervised_values, scaler)
        return test_transformed

    def diff_and_supervised(self, series_raw):
        # list of a = f(x) - f(x-1)
        diff_values = self.difference(series_raw)
        # supervised = [x, y] -> x = [window_len], y = [sequence_len]
        supervised = self.series_to_supervised(data=diff_values, n_in=self.window_len, n_out=self.sequence_len)
        return supervised.values

    def difference(self, series_raw):
        diff_series = self.difference_raw_values(series_raw.values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        return diff_values

    def difference_raw_values(self, raw_values_ndarray, interval=1):
        diff = list()
        for i in range(interval, len(raw_values_ndarray)):
            value = raw_values_ndarray[i] - raw_values_ndarray[i - interval]
            diff.append(value)
        return Series(diff)

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    def scale_train(self, train_supervised):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train_supervised)
        scaled_train = scaler.transform(train_supervised)
        return scaler, scaled_train

    def scale_test(self, test_supervised, scaler):
        scaled_values = scaler.transform(test_supervised)
        return scaled_values

    def split_and_reshape(self, ndarray):
        x, y = self.split_data(ndarray)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        return x, y

    def split_data(self, ndarray):
        return ndarray[:, :self.window_len], ndarray[:, self.window_len:]

    def reverse_transform(self, forecasts):
        test_remerged = self.reverse_split(forecasts)
        rescaled = self.scaler.inverse_transform(test_remerged)
        return rescaled[:, -self.sequence_len:]

    def reverse_split(self, forecasts):
        test_x = self.test_x
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1])
        test_remerged = []
        for i in range(test_x.shape[0]):
            row = [x for x in test_x[i]] + [y for y in forecasts[i]]
            row = np.array(row)
            row = row.reshape(1, len(row))
            test_remerged.append(row)
        test_remerged = np.array(test_remerged)
        test_remerged = test_remerged.reshape(test_remerged.shape[0], test_remerged.shape[2])
        return test_remerged

    def get_raw(self):
        return self.raw_split.train, self.raw_split.test

    def get_series_raw_values(self):
        return self.raw_series

    def get_raw_train_pandas_series(self):
        return self.raw_split.train

    def get_raw_test_pandas_series(self):
        return self.raw_split.test

    def get_train_transformed(self):
        return self.split_and_reshape(self.train_transformed)

    def get_test_transformed(self):
        return self.test_x, self.test_y
