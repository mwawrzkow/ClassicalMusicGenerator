from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from base_model import BaseModel
from tensorflow.keras.layers import LSTM, GRU, Dense, LeakyReLU, Bidirectional, BatchNormalization, Dropout, Input, Attention, Lambda
from tensorflow.keras.callbacks import TensorBoard
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

if not tf.config.list_physical_devices('GPU'):
    print("No GPU found, Training might be slow.")

@tf.keras.utils.register_keras_serializable()
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

class RNNModel(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_units=128, name="RNNModel"):
        super().__init__(input_dim, output_dim, name)
        self.hidden_units = hidden_units
        self.model = self.build_model()
        
    def build_model(self):
        inputs = Input(shape=(self.input_dim))
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(128)(x)
        outputs = {
        'pitch': Dense(128, name='pitch')(x),
        'step': Dense(1, name='step')(x),
        'duration': Dense(1, name='duration')(x),
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
    loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
    },
    optimizer=optimizer)
        model.summary()
        return model

    def train(self, dataset, epochs):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/fit/{timestamp}" 
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    tensorboard_callback
]
        self.model.fit(dataset, epochs=epochs,callbacks=callbacks, verbose=1)
    def load_checkpoint(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
    def predict_next_note(self,notes: np.ndarray,temperature: float = 1.0) -> tuple[int, float, float]:
        assert temperature > 0
        inputs = tf.expand_dims(notes, 0)
        predictions = self.model.predict(inputs, verbose=0)
        pitch_logits = predictions['pitch']
        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        step = predictions['step']
        duration = predictions['duration']
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)
        return int(pitch), float(step), float(duration)
    
    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        latest_checkpoint = tf.train.latest_checkpoint("./training_checkpoints/")
        if latest_checkpoint is None:
            print("No checkpoint found")
            return
        else :
            self.model.load_weights(latest_checkpoint)
            print (f"Model loaded from {latest_checkpoint}")
        print(f"Model loaded from {path}")