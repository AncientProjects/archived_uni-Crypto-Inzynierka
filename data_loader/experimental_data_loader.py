from base.base_data_loader import BaseDataLoader
from data.data_selector import DataSelector, Dataset
from data_loader.data_transformer.data_transformer import DataTransformer


class ExperimentalDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ExperimentalDataLoader, self).__init__(config)

        self.series_df = DataSelector.data_dict[Dataset.DEFAULT_BTC_DAY_2k]

        self.data_transformer = DataTransformer(self.series_df, self.config)
        self.train_raw, self.test_raw = self.data_transformer.get_raw()
        self.train_x, self.train_y = self.data_transformer.get_train_transformed()
        self.test_x, self.test_y = self.data_transformer.get_test_transformed()

    def get_test_plot_values(self):
        return self.test_raw.values

    def get_test_values_for_forecast(self):
        return self.data_transformer.get_series_raw_values()[-len(self.test_raw)-1:].values

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_test_values(self):
        return self.test_x, self.test_y

    def get_test_time(self):
        return self.data_transformer.test_time
