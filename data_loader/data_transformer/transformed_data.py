class TransformedData(object):
    def __init__(self, data_config, prepare_data):
        self.data_config = data_config
        self.scaler, self.train_transformed, self.test_transformed = prepare_data.transform_train_and_test_data()
        self.x_train_list, self.y_train_list = self.to_train_and_test(self.train_transformed)
        self.x_test_list, self.y_test_list = self.to_train_and_test(self.test_transformed)

    def to_train_and_test(self, list_kfold):
        x_list, y_list = [], []
        for arr in list_kfold:
            x, y = self.split_data(arr)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            x_list.append(x), y_list.append(y)
        return x_list, y_list

    def split_data(self, ndarray):
        window_len = self.data_config.window_size
        return ndarray[:, :window_len], ndarray[:, window_len:]
