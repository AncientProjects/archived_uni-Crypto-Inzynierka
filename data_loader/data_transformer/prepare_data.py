from pandas import Series, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

from data_loader.data_transformer.split_to_train_and_test import SplitToTrainAndTest


class PrepareData(object):
    def __init__(self, data_config, series_df):
        self.data_config = data_config
        self.close_values = series_df['close']
        self.time_values = series_df['time']

    def get_test_time(self):
        return self.time_values[-self.data_config.get_split():]

    # difference -> supervised -> fit scaler to train values and then transform test with it
    # save the scaler for test values
    def transform_train_and_test_data(self):
        # difference
        diff_series = self.diff_and_supervised(self.close_values)
        # supervised
        supervised_split = SplitToTrainAndTest(diff_series, self.data_config.get_split())
        # fit scaler and transform
        scaler, train_transformed = self.transform_train_data(supervised_split.train)
        test_transformed = self.transform_test_data(supervised_split.test, scaler)
        return scaler, train_transformed, test_transformed

    def diff_and_supervised(self, series_raw):
        # list of a = f(x) - f(x-1)
        diff_values = self.difference(series_raw)
        # supervised = [x, y] -> x = [window_len], y = [sequence_len]
        supervised = self.series_to_supervised(data=diff_values)
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

    def series_raw_supervised(self, series_raw):
        series_raw = series_raw.values
        series_raw = series_raw.reshape(len(series_raw), 1)
        result = self.series_to_supervised(series_raw)
        return result.values

    def series_to_supervised(self, data, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.data_config.window_size, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.data_config.sequence_len):
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

    def transform_train_data(self, train_supervised):
        # Series of tuples (index, value) to ndarray of values
        # scaler fit on supervised train ndarray of shape = (num_of_vectors, window_len + sequence_len)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_scaled = scaler.fit_transform(train_supervised)
        return scaler, train_scaled

    def transform_test_data(self, test_supervised, scaler):
        # Series of tuples (index, value) to ndarray of values
        # transform differenced and supervised test values using scaler fit to supervised train values
        test_scaled = scaler.transform(test_supervised)
        return test_scaled
