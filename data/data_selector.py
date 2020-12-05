from enum import Enum

from pandas import read_csv

from utils.utils import load_two_datasets


class Dataset(Enum):
    TESTERINO = 1
    TWO_DATASETS = 2
    SMALL_BTC = 3
    DEFAULT_BTC_DAY_2k = 4


class DataSelector(object):
    data_dict = {Dataset.TESTERINO: read_csv("data/testerino.csv"),
                 Dataset.TWO_DATASETS: load_two_datasets("data/bitcoin_1minute_1-1000.csv",
                                                         "data/bitcoin_1minute_1001-1201.csv"),
                 Dataset.SMALL_BTC: read_csv("data/BTCUSD100MINUT.csv"),
                 Dataset.DEFAULT_BTC_DAY_2k: read_csv("data/BTC-USD-2000-DAYkuj.csv")}
