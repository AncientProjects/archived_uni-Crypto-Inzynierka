from base.base_forecasts import BaseForecasts
from utils.utils import invert_data_transformations


class LSTMTestForecasts(BaseForecasts):
    def __init__(self, model, data_loader, config):
        super(LSTMTestForecasts, self).__init__(data_loader, config)
        self.model = model
        self.batch_size = self.config.trainer.batch_size
        #       tuple -> self.X_test = self.test_data[0], self.y_test = self.test_data[1]
        self.test_data = self.data_loader.get_test_data()
        self.forecasts_rescaled = []

    def forecast_with_model(self):
        self.forecasts_scaled = self.model.predict(
            self.test_data[0],
            batch_size=self.config.trainer.batch_size,
            verbose=self.config.trainer.verbose_training
        )
        return self.forecasts_scaled
        # return self.inverse_transform_forecasts()

    def forecast_without_model(self):
        print("LSTM must have a model!")

    def inverse_transform_forecasts(self):
        self.forecasts_rescaled = invert_data_transformations(self.test_data,
                                                              self.data_loader.get_test_values(),
                                                              self.forecasts_scaled,
                                                              self.data_loader.scaler)
        return self.forecasts_rescaled
