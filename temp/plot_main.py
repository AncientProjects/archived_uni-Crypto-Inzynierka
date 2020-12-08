import pandas as pd

from data.data_selector import DataSelector, Dataset
from utils.plot import plot_series


def main():
    series_df = DataSelector.data_dict[Dataset.DEFAULT_BTC_DAY_2k]
    time_series = pd.to_datetime(series_df['time'], unit='s')
    close_series = series_df['close'].values
    plot_series(time_series, close_series)


if __name__ == '__main__':
    main()