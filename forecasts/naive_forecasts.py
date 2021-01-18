from math import sqrt

from sklearn.metrics import mean_squared_error

from base.base_forecasts import BaseForecasts


class NaiveForecasts(BaseForecasts):
    def __init__(self, data_loader, config):
        super(NaiveForecasts, self).__init__(data_loader, config)

    def forecast(self, test_raw, test_data, i):
        print("Naive forecast doesn't have a model")

    def forecast_without_model(self, test_raw, i):
        forecasts = test_raw[:-1]
        test_raw = test_raw[1:]
        rmse = self.evaluate(test_raw, forecasts)
        forecasts = forecasts.reshape(len(forecasts), self.config.data_loader.sequences)
        return test_raw, forecasts, rmse

    def evaluate(self, test_raw, test_calculated):
        forecasts = test_calculated.flatten()
        rmse = sqrt(mean_squared_error(test_raw, forecasts))
        return rmse
