from utils.config import process_config
from utils import utils
from utils import factory
from utils import plot
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
keras = tf.keras


def main():
    try:
        args = utils.get_args()
        config = process_config(args.config)

        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        data_loader = factory.create("data_loader." + config.data_loader.name)(config)

        if config.exp.type == "full":
            model = factory.create("models." + config.model.name)(data_loader.get_train_data(), config)
            model.load_model()

            model_forecasts = factory.create("forecasts." + config.forecasts.name)(model.model, data_loader, config)
            test_raw, forecasts = model_forecasts.forecast_with_model()
            time_datetime = pd.to_datetime(data_loader.get_test_time(), unit='s')
            plot.plot_real_and_forecasts(time_datetime, test_raw, forecasts)
            # plot.plot_real_and_forecastsv2(data_loader.get_test_time(), data_loader.get_test_values(), forecasts)

            # asd = data_loader.get_test_values()
            # asd = asd[1]
            # asd = asd[:, 0]
            # dsa = forecasts[:, 0]

            # print(keras.metrics.mean_absolute_error(asd, dsa).numpy())
        else:
            model_forecasts = factory.create("forecasts." + config.forecasts.name)(data_loader, config)
            forecasts = model_forecasts.forecast_without_model()

            plot.plot_no_model_results(data_loader.time_test, data_loader.x_test, forecasts, config.exp.name)

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
