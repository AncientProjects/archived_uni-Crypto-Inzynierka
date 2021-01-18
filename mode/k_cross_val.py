import tensorflow as tf

import utils

keras = tf.keras


class CrossVal(object):
    def __init__(self, config):
        self.config = config
        self.error_scores = []

    def execute(self):
        keras.backend.clear_session()
        data_loader = utils.factory.create("data_loader." + self.config.data_loader.name)(self.config)

        model = utils.factory.create("models." + self.config.model.name)(self.config)

        for o in range(0, 5):
            for i in range(0, self.config.data_loader.k_fold):
                trainer = utils.factory.create("trainers." + self.config.trainer.name)(model.model,
                                                                                       self.config)
                history, trained_model = trainer.train(data_loader.get_train_data(i), data_loader.get_test_data(i))

                model_forecasts = utils.factory.create("forecasts." + self.config.forecasts.name)(trained_model,
                                                                                                  data_loader,
                                                                                                  self.config)
                test_raw, forecasts, rmse = model_forecasts.forecast(data_loader.get_test_raw_values(i),
                                                                     data_loader.get_test_data(i), i)
                self.error_scores.append(rmse)

        utils.plot.plot_score_boxplot(self.error_scores)

        # test_raw, forecasts = model_forecasts.forecast_with_model()
        # time_datetime = pd.to_datetime(data_loader.get_test_time(), unit='s')
        # plot.plot_real_and_forecasts(time_datetime, test_raw, forecasts)
