# from pandas_datareader import data
# import datetime
from math import sqrt
from numpy import cumsum, std, subtract, polyfit, log
# from numpy.random import randn


def hurst(ts):
    lags = range(2, 100)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    return poly[0] * 2.0
