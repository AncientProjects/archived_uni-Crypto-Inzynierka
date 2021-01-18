import numpy as np

from data_loader.data_transformer.data_config import DataConfig
from data_loader.data_transformer.prepare_data import PrepareData
from data_loader.data_transformer.split_to_train_and_test import SplitToTrainAndTest
from data_loader.data_transformer.transformed_data import TransformedData


class DataTransformer(object):
    def __init__(self, series_df, config):
        self.data_config = DataConfig(config, len(series_df))
        self.prepare_data = PrepareData(self.data_config, series_df)

        self.raw_series = self.prepare_data.close_values

        # raw_train and raw_test Series with real values from data before any transformations
        self.raw_split = SplitToTrainAndTest(self.raw_series, self.data_config)

        self.transformed_data = TransformedData(self.data_config, self.prepare_data)

    def reverse_transform(self, forecasts, x_test, i):
        test_remerged = self.reverse_split(forecasts, x_test)
        rescaled = self.transformed_data.scaler[i].inverse_transform(test_remerged)
        if self.data_config.recursive:
            n_seq = self.data_config.recursive_seq
        else:
            n_seq = self.data_config.sequence_len
        return rescaled[:, -n_seq:]

    def reverse_split(self, forecasts, x_test):
        test_x = x_test
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

    def get_raw_test(self, i):
        return self.raw_split.test_kfold[i]

    def get_series_raw_values(self):
        return self.raw_series
