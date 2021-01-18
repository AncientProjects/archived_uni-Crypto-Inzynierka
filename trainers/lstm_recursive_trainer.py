import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow_core.python.keras.callbacks import LearningRateScheduler

from base.base_trainer import BaseTrain


class LSTMRecursiveTrainer(BaseTrain):
    def __init__(self, model, config):
        super(LSTMRecursiveTrainer, self).__init__(model, config)
        self.callbacks = []
        self.history = {}
        self.init_callbacks()

    def init_callbacks(self):
        if self.config.trainer.model_checkpoint:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                          '%s-{epoch:02d}-{val_loss:.5f}.hdf5' % self.config.exp.name),
                    monitor=self.config.callbacks.checkpoint_monitor,
                    mode=self.config.callbacks.checkpoint_mode,
                    save_best_only=self.config.callbacks.checkpoint_save_best_only,
                    save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                    verbose=self.config.callbacks.checkpoint_verbose
                )
            )
        if self.config.trainer.tensorboard:
            self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.callbacks.tensorboard_log_dir,
                    write_graph=self.config.callbacks.tensorboard_write_graph,
                )
            )
        if self.config.trainer.early_stopping:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.callbacks.early_stopping_patience
                )
            )
        if self.config.trainer.lr_scheduler:
            self.callbacks.append(
                LearningRateScheduler(
                    lambda epoch: 1e-8 * 10 ** (epoch / 20))
            )

    def train(self, train_data, test_data):
        history = self.model.fit(
            train_data[0], train_data[1][:, 0:1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            # batch_size=self.train_data[0].shape[0] // 5,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        return history, self.model
