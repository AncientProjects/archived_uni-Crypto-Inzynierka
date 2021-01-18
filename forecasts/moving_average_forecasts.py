from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

from base.base_forecasts import BaseForecasts


class MovingAverageForecasts(BaseForecasts):
    def __init__(self, data_loader, config):
        super(MovingAverageForecasts, self).__init__(data_loader, config)

    def forecast(self, test_raw, test_data, i):
        pass

    def forecast_without_model(self, test_raw, i):
        n_lag = self.config.data_loader.window_size
        n_seq = self.config.data_loader.sequences
        list_avg = list()
        for j in range(n_lag, len(test_raw) - n_seq):
            sum = 0
            for o in range(0, n_lag + n_seq):
                sum += test_raw[j - n_lag + o]
            avg = sum / (n_lag + n_seq)
            list_avg.append(avg)
        forecasts = np.array(list_avg)
        test_raw = test_raw[n_lag:-n_seq]

        rmse = sqrt(mean_squared_error(test_raw, forecasts))
        forecasts = forecasts.reshape(len(forecasts), 1)
        return test_raw, forecasts, rmse
