import numpy as np

from data_loader.data_transformer.data_config import DataConfig
from data_loader.data_transformer.prepare_data import PrepareData
from data_loader.data_transformer.split_to_train_and_test import SplitToTrainAndTest
from data_loader.data_transformer.transformed_data import TransformedData


class DataTransformer(object):
    def __init__(self, series_df, config):
        self.data_config = DataConfig(config)
        self.prepare_data = PrepareData(self.data_config, series_df)

        self.raw_series = self.prepare_data.close_values
        self.test_time = self.prepare_data.get_test_time()

        # raw_train and raw_test Series with real values from data before any transformations
        self.raw_split = SplitToTrainAndTest(self.raw_series, self.data_config.get_split())

        self.transformed_data = TransformedData(self.data_config, self.prepare_data)

        test_actual = self.prepare_data.series_raw_supervised(self.raw_series[-self.data_config.test_len:])
        _, self.y_test_actual = self.transformed_data.split_data(test_actual)

    def reverse_transform(self, forecasts):
        test_remerged = self.reverse_split(forecasts)
        rescaled = self.transformed_data.scaler.inverse_transform(test_remerged)
        return rescaled[:, -self.data_config.sequence_len:]

    def reverse_split(self, forecasts):
        test_x = self.transformed_data.x_test
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
        return self.transformed_data.get_train_data()

    def get_test_transformed(self):
        return self.transformed_data.get_test_data()
