class BaseForecasts(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
        self.forecasts_scaled = []
        self.window_size = self.config.data_loader.window_size
        self.sequence_size = self.config.data_loader.sequences

    def forecast(self, test_raw, test_data, i):
        raise NotImplementedError

    def forecast_without_model(self, test_raw, i):
        raise NotImplementedError
