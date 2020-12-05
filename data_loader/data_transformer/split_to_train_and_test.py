

class SplitToTrainAndTest(object):
    def __init__(self, series, split):
        self.train = series[:-split]
        self.test = series[-split:]
