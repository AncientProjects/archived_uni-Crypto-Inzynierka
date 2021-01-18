import sys

import tensorflow as tf

from mode import forecast
from mode import k_cross_val
from mode import train
from utils import utils
from utils.config import process_config

keras = tf.keras


def set_dataset(args):
    if args.dataset == '1':
        return "btc_min"
    if args.dataset == '2':
        return "btc_day"


def main():
    try:
        args = utils.get_args()
        config = process_config(args.config)
        config.data_loader.dataset = set_dataset(args)

        if args.mode == 'train':
            train.Train(config, args).train()
        elif args.mode == 'forecast':
            forecast.Forecast(config).forecast()
        else:
            k_cross_val.CrossVal(config).execute()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
