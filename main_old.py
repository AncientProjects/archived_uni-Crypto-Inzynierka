from utils.config import process_config
from utils import utils
from utils.dirs import create_dirs
from utils import factory
from utils import plot
import sys
import tensorflow as tf
keras = tf.keras


def main():
    try:
        args = utils.get_args()
        config = process_config(args.config)

        data_loader = factory.create("data_loader." + config.data_loader.name)(config)
        if config.exp.type == "full":
            create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])
            model = factory.create("models." + config.model.name)(data_loader.get_train_data(), config)

            trainer = factory.create("trainers." + config.trainer.name)(model.model,
                                                                        data_loader.get_train_data(),
                                                                        data_loader.get_test_data(), config)
            trainer.train_with_early_stopping()

            forecasts = factory.create("forecasts." + config.forecasts.name)(trainer.model, data_loader, config)
            forecast = forecasts.forecast_with_model()

            plot.plot_real_and_forecasts(data_loader.time_test, data_loader.x_test, forecast)
        else:
            forecasts = factory.create("forecasts." + config.forecasts.name)(data_loader, config)
            forecast = forecasts.forecast_without_model()

            plot.plot_no_model_results(data_loader.time_test, data_loader.x_test, forecast, config.exp.name)

        print(keras.metrics.mean_absolute_error(data_loader.x_test, forecast).numpy())

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
