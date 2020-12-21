from math import sqrt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

import utils
from utils import plot

keras = tf.keras


class Forecast(object):
    def __init__(self, config):
        self.config = config
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

    def forecast(self):
        data_loader = utils.factory.create("data_loader." + self.config.data_loader.name)(self.config)

        model = utils.factory.create("models." + self.config.model.name)(data_loader.get_train_data(), self.config)
        model.load_model()

        model_forecasts = utils.factory.create("forecasts." + self.config.forecasts.name)(model.model, data_loader,
                                                                                          self.config)
        test_raw, forecasts = model_forecasts.forecast_with_model()
        time_datetime = pd.to_datetime(data_loader.get_test_time(), unit='s')
        plot.plot_real_and_forecasts(time_datetime, test_raw, forecasts)
        print('RMSE: ', self.evaluate(test_raw, forecasts))

    def evaluate(self, test_raw, forecasts):
        n_seq = self.config.data_loader.sequences
        if n_seq == 1:
            print('RMSE: ', sqrt(mean_squared_error(test_raw, forecasts)))
        for i in range(n_seq):
            actual = [row[i] for row in test_raw]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))
