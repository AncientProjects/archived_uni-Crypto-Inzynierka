from enum import Enum

from pandas import read_csv


class Dataset(Enum):
    TESTERINO = 1
    TWO_DATASETS = 2
    SMALL_BTC = 3
    DEFAULT_BTC_DAY_2k = 4


class DataSelector(object):
    testerino = "data/testerino.csv"
    data_btc1 = "data/bitcoin_1minute_1-1000.csv"
    data_btc2 = "data/bitcoin_1minute_1001-1201.csv"
    small_btc = "data/BTCUSD100MINUT.csv"
    default_btc = "data/BTC-USD-2000-DAYkuj.csv"

    data_dict = {
        Dataset.TESTERINO: read_csv(testerino),
        Dataset.TWO_DATASETS: (read_csv(data_btc1), read_csv(data_btc2)),
        Dataset.SMALL_BTC: read_csv(small_btc),
        Dataset.DEFAULT_BTC_DAY_2k: read_csv(default_btc)
    }
