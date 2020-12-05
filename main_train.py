import sys

import numpy as np
import tensorflow as tf

from utils import factory
from utils import utils
from utils.config import process_config
from utils.dirs import create_dirs

keras = tf.keras


def main():
    try:
        args = utils.get_args()
        config = process_config(args.config)

        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        data_loader = factory.create("data_loader." + config.data_loader.name)(config)

        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])
        model = factory.create("models." + config.model.name)(data_loader.get_train_data(), config)

        trainer = factory.create("trainers." + config.trainer.name)(model.model, data_loader.get_train_data(), config)
        # trainer.train_with_early_stopping()
        history = trainer.train()

        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([1e-8, 1e-2, 0, 0.03])
        # plt.show()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
