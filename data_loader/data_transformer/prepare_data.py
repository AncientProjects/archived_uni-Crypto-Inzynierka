
class PrepareData(object):
    def __init__(self, config, series_df):
        self.config = config
        self.close_values = series_df['close']
        self.time_values = series_df['time']


