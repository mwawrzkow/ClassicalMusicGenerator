import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from tensorflow.keras import Model
import numpy as np
from base_model import BaseModel

@tf.keras.utils.register_keras_serializable()
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def positional_encoding(self, length, depth):
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class TransformerDecoder(Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransformerModel(BaseModel):
    def __init__(self, input_dim, output_dim, d_model=128, num_heads=8, dff=512, num_layers=4, rate=0.1, name="TransformerModel"):
        super().__init__(input_dim, output_dim, name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.rate = rate
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.input_dim[0],), dtype=tf.int32)

        # We only need the pitch for the embedding
        pitch_input = inputs

        self.pos_embedding = PositionalEmbedding(vocab_size=128, d_model=self.d_model)
        x = self.pos_embedding(pitch_input)

        self.dec_layers = [TransformerDecoder(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training=True, mask=self.create_causal_mask(x))

        outputs = {
            'pitch': Dense(128, name='pitch')(x),
            'step': Dense(1, name='step')(x),
            'duration': Dense(1, name='duration')(x),
        }

        model = Model(inputs=inputs, outputs=outputs)

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
                'duration': 1.0,
            },
            optimizer=optimizer)

        model.summary()
        return model

    def create_causal_mask(self, x):
        seq_len = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def train(self, dataset, epochs):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints_transformer/ckpt_{epoch}',
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True)
        ]
        self.model.fit(dataset, epochs=epochs, callbacks=callbacks)

    def predict_next_note(self, notes: np.ndarray, temperature: float = 1.0) -> tuple[int, float, float]:
        assert temperature > 0

        # The model expects a batch of sequences.
        inputs = tf.expand_dims(notes, 0)

        # The model returns logits for the last note in the sequence.
        predictions = self.model.predict(inputs)

        pitch_logits = predictions['pitch'][:, -1, :]
        step_logits = predictions['step'][:, -1, :]
        duration_logits = predictions['duration'][:, -1, :]

        pitch_logits /= temperature

        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)

        step = tf.maximum(0, step_logits)
        duration = tf.maximum(0, duration_logits)

        step = tf.squeeze(step, axis=-1)
        duration = tf.squeeze(duration, axis=-1)

        return int(pitch[0]), float(step[0]), float(duration[0])

    def save(self, path):
        self.model.save_weights(path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_weights(path)
        print(f"Model loaded from {path}")
