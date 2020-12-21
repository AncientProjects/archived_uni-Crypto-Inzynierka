import numpy as np
import tensorflow as tf

import utils

keras = tf.keras


class Train(object):
    def __init__(self, config):
        self.config = config
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

    def train(self):
        data_loader = utils.factory.create("data_loader." + self.config.data_loader.name)(self.config)

        utils.dirs.create_dirs([self.config.callbacks.tensorboard_log_dir, self.config.callbacks.checkpoint_dir])
        model = utils.factory.create("models." + self.config.model.name)(data_loader.get_train_data(), self.config)

        trainer = utils.factory.create("trainers." + self.config.trainer.name)(model.model, data_loader.get_train_data(), self.config)
        history = trainer.train()
