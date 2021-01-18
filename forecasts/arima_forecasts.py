from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

from base.base_forecasts import BaseForecasts


class ArimaForecasts(BaseForecasts):
    def __init__(self, data_loader, config):
        super(ArimaForecasts, self).__init__(data_loader, config)

    def forecast(self, test_raw, test_data, i):
        pass

    def forecast_without_model(self, test_raw, i):
        window_size = self.config.data_loader.window_size
        train = self.data_loader.get_train_raw_values(i)
        history = [x for x in train]
        forecasts = list()
        # walk-forward validation
        for t in range(len(test_raw)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            forecasts.append(yhat)
            obs = test_raw[t]
            history.append(obs)
        forecasts = np.array(forecasts)
        rmse = self.evaluate(test_raw, forecasts)
        forecasts = forecasts.reshape(len(forecasts), self.config.data_loader.sequences)
        return test_raw, forecasts, rmse

    def evaluate(self, test_raw, test_calculated):
        forecasts = test_calculated.flatten()
        rmse = sqrt(mean_squared_error(test_raw, forecasts))
        return rmse
