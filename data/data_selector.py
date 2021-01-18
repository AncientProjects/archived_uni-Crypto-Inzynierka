from pandas import read_csv


class DataSelector(object):
    default_btc = "data/BTC-USD-2000-DAY.csv"
    btc_min_2k = "data/BTC-USD-2000-MINUT.csv"

    data_dict = {
        "btc_day": read_csv(default_btc),
        "btc_min": read_csv(btc_min_2k)
    }
