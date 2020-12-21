import sys

import tensorflow as tf

from mode import forecast
from mode import train
from utils import utils
from utils.config import process_config

keras = tf.keras


def main():
    try:
        args = utils.get_args()
        config = process_config(args.config)

        if args.mode == 'train':
            train.Train(config).train()
        elif args.mode == 'forecast':
            forecast.Forecast(config).forecast()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
