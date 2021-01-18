from base.base_data_loader import BaseDataLoader
from data.data_selector import DataSelector
from data_loader.data_transformer.data_transformer import DataTransformer


class LSTMSingleOutputDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(LSTMSingleOutputDataLoader, self).__init__(config)

        self.series_df = DataSelector.data_dict[self.config.data_loader.dataset]

        self.data_transformer = DataTransformer(self.series_df, config)

    def get_train_raw_values(self, i):
        return self.data_transformer.raw_split.train_kfold[i]

    def get_test_raw_values(self, i):
        return self.data_transformer.raw_split.test_kfold[i]

    def get_train_data(self, i):
        return self.data_transformer.transformed_data.x_train_list[i], \
               self.data_transformer.transformed_data.y_train_list[i]

    def get_test_data(self, i):
        return self.data_transformer.transformed_data.x_test_list[i], \
               self.data_transformer.transformed_data.y_test_list[i]

    def get_test_time_values(self, i):
        time_values = self.series_df['time'].values
        indexes = self.data_transformer.raw_split.test_indexes[i]
        return time_values[indexes]

    def get_train_and_test_raw_set(self, i):
        values = self.series_df['close'].values
        return values[self.data_transformer.raw_split.train_indexes[i]], values[
            self.data_transformer.raw_split.test_indexes[i]]
