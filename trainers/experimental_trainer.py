import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from base.base_trainer import BaseTrain


class ExperimentalTrainer(BaseTrain):
    def __init__(self, model, train_data, config):
        super(ExperimentalTrainer, self).__init__(model, train_data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.history = {}
        self.init_callbacks()

    def init_callbacks(self):
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

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                patience=self.config.callbacks.early_stopping_patience
            )
        )

        # self.callbacks.append(
        #     LearningRateScheduler(
        #         lambda epoch: 1e-8 * 10 ** (epoch / 20))
        # )

    def train(self):
        return self.model.fit(
            self.train_data[0], self.train_data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            # batch_size=self.config.trainer.batch_size,
            batch_size=self.train_data[0].shape[0] // 5,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
