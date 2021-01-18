from math import sqrt

import numpy as np

from base.base_forecasts import BaseForecasts


class NaiveSequenceForecasts(BaseForecasts):
    def __init__(self, data_loader, config):
        super(NaiveSequenceForecasts, self).__init__(data_loader, config)

    def forecast(self, test_raw, test_data, i):
        print("Naive forecast doesn't have a model")

    def forecast_without_model(self, test_raw, i):
        n_seq = self.config.data_loader.sequences
        n_lag = self.config.data_loader.window_size
        forecasts, actual = self.calculate_naive(test_raw, n_lag, n_seq)
        rmse = self.evaluate(actual, forecasts)
        forecasts = forecasts[:, n_lag:]
        return test_raw, forecasts, rmse

    def calculate_naive(self, test_raw, n_lag, n_seq):
        forecasts = list()
        actual = list()
        for i in range(0, len(test_raw) - n_seq):
            row_forecast = list()
            row_actual = list()
            for o in range(0, n_lag + n_seq):
                row_forecast.append(test_raw[i])
                row_actual.append(test_raw[i + o])
            forecasts.append(row_forecast)
            actual.append(row_actual)
        return np.array(forecasts), np.array(actual)

    def evaluate(self, actual, forecasts):
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - forecasts[row, col]) ** 2
        rmse = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return rmse
