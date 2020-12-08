import numpy as np
import pandas as pd
import tensorflow as tf

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

        model_forecasts = utils.factory.create("forecasts." + self.config.forecasts.name)(model.model, data_loader, self.config)
        test_raw, forecasts = model_forecasts.forecast_with_model()
        time_datetime = pd.to_datetime(data_loader.get_test_time(), unit='s')
        plot.plot_real_and_forecasts(time_datetime, test_raw, forecasts)
