import argparse

from pandas import read_csv


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-m', '--mode',
        dest='mode',
        metavar='M',
        default='None',
        help='Mode can be train or forecast')
    args = argparser.parse_args()
    return args


def load_two_datasets(training_data_path, validation_data_path):
    series = read_csv(training_data_path)
    series_val = read_csv(validation_data_path)
    return series, series_val

