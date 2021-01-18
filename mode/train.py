import numpy as np
import tensorflow as tf

import utils

keras = tf.keras


class Train(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        if args.callbacks == 'lr':
            self.config.trainer.model_checkpoint = False
            self.config.trainer.tensorboard = True
            self.config.trainer.early_stopping = False
            self.config.trainer.lr_scheduler = True
            self.config.trainer.num_epochs = 100
        else:
            self.config.trainer.model_checkpoint = True

    def train(self):
        data_loader = utils.factory.create("data_loader." + self.config.data_loader.name)(self.config)

        utils.dirs.create_dirs([self.config.callbacks.tensorboard_log_dir, self.config.callbacks.checkpoint_dir])
        model = utils.factory.create("models." + self.config.model.name)(self.config)

        trainer = utils.factory.create("trainers." + self.config.trainer.name)(model.model, self.config)
        i = self.config.data_loader.k_fold - 1
        history, trained_model = trainer.train(data_loader.get_train_data(i), data_loader.get_test_data(i))

        if self.args.callbacks == 'lr':
            utils.plot.plot_lr_by_loss2(history)
        else:
            utils.plot.plot_learning_curves(history)

        # plot_data = data_loader.get_train_and_test_raw_set(i)
        # utils.plot.plot_series_with_ticks(plot_data[0], y_label='train_set')
        # utils.plot.plot_series_with_ticks(plot_data[1], y_label='test_set')

    def mean_error_score(self, history):
        loss = history['loss']
        val_loss = history['val_loss']
        loss = loss[10:70]
        val_loss = val_loss[10:70]
        print('Average loss: %.3f, val_loss: %.3f ', np.average(loss), np.average(val_loss))
