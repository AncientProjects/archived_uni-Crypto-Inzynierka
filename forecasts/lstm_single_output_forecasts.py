from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

from base.base_forecasts import BaseForecasts


class LSTMSingleOutputForecasts(BaseForecasts):
    def __init__(self, model, data_loader, config):
        super(LSTMSingleOutputForecasts, self).__init__(data_loader, config)
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
        # test_calculated = self.calculate_predicted(test_raw, test_diff)
        # test_raw = test_raw[1:]
        test_calculated = test_diff
        rmse = self.evaluate(test_raw, test_calculated)
        return test_raw, test_calculated, rmse

    def calculate_predicted(self, test_raw, test_diff):
        test_calculated = []
        for i in range(0, len(test_diff)):
            forecast = test_raw[i] + test_diff[i]
            test_calculated.append(forecast)
        test_calculated = np.array(test_calculated)
        return test_calculated

    def evaluate(self, test_raw, test_calculated):
        forecasts = test_calculated.flatten()
        rmse = sqrt(mean_squared_error(test_raw, forecasts))
        return rmse

    def forecast_without_model(self, test_raw, i):
        print("LSTM must have a model!")
