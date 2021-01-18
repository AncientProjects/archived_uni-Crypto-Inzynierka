import pandas as pd
import tensorflow as tf

import utils
from utils import plot

keras = tf.keras


class Forecast(object):
    def __init__(self, config):
        self.config = config
        keras.backend.clear_session()
        self.error_score = None
        # tf.random.set_seed(42)
        # np.random.seed(42)

    def forecast(self):
        data_loader = utils.factory.create("data_loader." + self.config.data_loader.name)(self.config)
        if self.config.exp.type == 'full':
            model = utils.factory.create("models." + self.config.model.name)(self.config)
            model.load_model()

            model_forecasts = utils.factory.create("forecasts." + self.config.forecasts.name)(model.model, data_loader,
                                                                                              self.config)
            i = self.config.data_loader.k_fold - 1
            test_raw, forecasts, self.error_score = model_forecasts.forecast(data_loader.get_test_raw_values(i),
                                                                             data_loader.get_test_data(i), i)
        else:
            model_forecasts = utils.factory.create("forecasts." + self.config.forecasts.name)(data_loader, self.config)

            i = 4
            test_raw, forecasts, self.error_score = model_forecasts.forecast_without_model(
                data_loader.get_test_raw_values(i), i)

        self.plot(data_loader, forecasts, i, data_loader.get_test_raw_values(i))
        print('RMSE: ', self.error_score)

    def plot(self, data_loader, forecasts, i, test_raw):
        index = 100
        time_datetime = pd.to_datetime(data_loader.get_test_time_values(i), unit='s')
        time_datetime = time_datetime[self.config.data_loader.window_size:]
        time_datetime, test_raw, forecasts = time_datetime[-index:], test_raw[-index:], forecasts[-index:]
        plot.plot_real_and_forecasts(time_datetime, test_raw, forecasts)
