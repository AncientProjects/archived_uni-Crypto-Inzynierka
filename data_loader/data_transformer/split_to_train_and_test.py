import numpy as np
from sklearn.model_selection import KFold


class SplitToTrainAndTest(object):
    def __init__(self, series, data_config):
        self.series = series
        self.k = data_config.k
        self.train_kfold, self.test_kfold, self.train_indexes, self.test_indexes = self.fold()

    def fold(self):
        series_array = np.array(self.series)
        kfold = KFold(self.k)
        train_kfold, test_kfold = list(), list()
        test_indexes, train_indexes = list(), list()
        for train, test in kfold.split(series_array):
            train_kfold.append(series_array[train])
            test_kfold.append(series_array[test])
            test_indexes.append(test)
            train_indexes.append(train)
        train_kfold, test_kfold = np.array(train_kfold), np.array(test_kfold)
        return train_kfold, test_kfold, train_indexes, test_indexes
