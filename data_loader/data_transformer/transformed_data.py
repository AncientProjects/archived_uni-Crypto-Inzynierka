class TransformedData(object):
    def __init__(self, data_config, prepare_data):
        self.data_config = data_config
        self.scaler, self.train_transformed, self.test_transformed = prepare_data.transform_train_and_test_data()
        self.x_train, self.y_train = self.split_and_reshape(self.train_transformed)
        self.x_test, self.y_test = self.split_and_reshape(self.test_transformed)

    def split_and_reshape(self, ndarray):
        x_train, y_train = self.split_data(ndarray)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        return x_train, y_train

    def split_data(self, ndarray):
        window_len, _ = self.data_config.get_window_and_sequence()
        return ndarray[:, :window_len], ndarray[:, window_len:]

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test
