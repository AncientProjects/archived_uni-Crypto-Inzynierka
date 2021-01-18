import argparse


def get_args():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    arg_parser.add_argument(
        '-m', '--mode',
        dest='mode',
        metavar='M',
        default='None',
        help='Mode can be train or forecast')
    arg_parser.add_argument(
        '-cb', '--callbacks',
        dest='callbacks',
        metavar='CB',
        default='None',
        help='Change callbacks used')
    arg_parser.add_argument(
        '-d', '--dataset',
        dest='dataset',
        metavar='D',
        default='1',
        help='Choose dataset; 1 - minute timestep, 2 - day timestep')
    args = arg_parser.parse_args()
    return args
