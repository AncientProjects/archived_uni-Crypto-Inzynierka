import tensorflow
from tensorflow.keras.models import load_model

from base.base_model import BaseModel

tf = tensorflow
keras = tf.keras


class LSTMSingleOutputModel3(BaseModel):
    def __init__(self, config):
        super(LSTMSingleOutputModel3, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.LSTM(80, input_shape=(self.window_size, 1), activation='relu', return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(40, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(self.sequence_size, activation='relu'))
        # self.model.add(tf.keras.layers.Lambda(lambda x: x * 200.0))

        self.model.compile(
            # loss=keras.losses.Huber(),
            loss='mse',
            optimizer=self.config.model.optimizer,
            # optimizer=keras.optimizers.SGD(lr=1e-8, momentum=0.9),
            # optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-8),
            metrics=['mae'],
        )
        print(self.model.summary())

    def load_model(self):
        self.model = load_model(
            filepath=self.config.model.load_model_path,
            custom_objects=None, compile=True)

# optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.momentum, epsilon=1e-10),
# optimizer=keras.optimizers.SGD(self.learning_rate, self.momentum),
