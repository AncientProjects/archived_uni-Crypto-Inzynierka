from math import sqrt

import numpy as np
from pandas import DataFrame, concat

from base.base_forecasts import BaseForecasts


class LSTMMultiOutputForecasts(BaseForecasts):
    def __init__(self, model, data_loader, config):
        super(LSTMMultiOutputForecasts, self).__init__(data_loader, config)
        self.model = model
        #       tuple -> self.X_test = self.test_data[0], self.y_test = self.test_data[1]
        self.forecasts_rescaled = []

    def forecast(self, test_raw, test_data, i):
        self.forecasts_scaled = self.model.predict(
            test_data[0],
            batch_size=self.config.trainer.batch_size,
            verbose=self.config.trainer.verbose_training
        )
        test_diff = self.data_loader.data_transformer.reverse_transform(self.forecasts_scaled, test_data[0], i)
        test_raw = test_raw[self.config.data_loader.window_size:]
        test_calculated = self.calculate_predicted(test_raw, test_diff)
        test_raw = self.supervised(test_raw.reshape(len(test_raw), 1))
        test_raw = test_raw.values
        rmse = self.evaluate(test_raw, test_calculated)
        return test_raw, test_calculated, rmse

    def calculate_predicted(self, test_raw, test_diff):
        test_calculated = []
        for i in range(0, len(test_diff)):
            temp = []
            diff = test_diff[i]
            forecast = test_raw[i]
            for o in range(0, self.sequence_size):
                forecast = diff[o]
                temp.append(forecast)
            test_calculated.append(temp)
        test_calculated = np.array(test_calculated)
        return test_calculated

    def evaluate(self, test_raw, test_calculated):
        s = 0
        for row in range(test_raw.shape[0]):
            for col in range(test_raw.shape[1]):
                s += (test_raw[row, col] - test_calculated[row, col]) ** 2
        rmse = sqrt(s / (test_raw.shape[0] * test_raw.shape[1]))
        return rmse

    def supervised(self, data, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)

        cols, names = list(), list()
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.config.data_loader.sequences):
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

    def forecast_without_model(self, test_raw, i):
        print("LSTM must have a model!")
