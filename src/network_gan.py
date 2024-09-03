import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, GRU, LeakyReLU, Bidirectional
from tensorflow.keras.optimizers.schedules import ExponentialDecay



# test also with tensorflow.python.keras 
# Check LSTM without dense 
# simple RNN 
# smaller dataset bigger batch sizes 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tqdm import tqdm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.profiler import profiler_v2 as profiler
from datetime import datetime
from tensorflow.python.client import device_lib
from base_model import BaseModel
# tf.debugging.set_log_device_placement(True)
print(device_lib.list_local_devices())
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("No GPU found, can't continue.")

# enable jit compilation
tf.config.optimizer.set_jit("autoclustering")
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# tf.config.optimizer.set_jit('autoclustering')
logdir = None
tensorboard_callback = None
@tf.keras.utils.register_keras_serializable()
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)
class GAN(BaseModel):
    """
    Generative Adversarial Network (GAN) class.
    """

    def __init__(self, input_dim, generator_output_dim, discriminator_output_dim, load=False):
        """
        Constructor for the GAN class.

        Parameters:
        - input_dim (int): Dimension of the input noise vector.
        - generator_output_dim (int): Dimension of the generator output.
        - discriminator_output_dim (int): Dimension of the discriminator output.
        """
        # print available devices
        print(f"Available devices: {tf.config.list_physical_devices('GPU')}")
        # if no GPU is available, crash
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError("No GPU found, can't continue.")

        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim
        self.discriminator_output_dim = discriminator_output_dim
        # if load is True, load the model
        self.load = load
        if load:
            self.generator = tf.keras.models.load_model("./generator.keras")
            self.critic = tf.keras.models.load_model("./discriminator.keras")
            print("Model loaded successfully.")
            self.generator.summary()
            self.critic.summary()
            # compile the model after loading
            # self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            # self.generator.compile(optimizer=self.generator_optimizer)
            # self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            # self.critic.compile(optimizer=self.critic_optimizer)
            

        else: 
            self.generator = self.build_generator()
            self.critic = self.build_discriminator()

        
    def build_generator(self):
        inputs = Input(shape=(self.input_dim))
        
        # LSTM with return_sequences=True to maintain sequence length
        x = LSTM(256, return_sequences=True)(inputs)  # Ensure return_sequences=True to produce a sequence
        x = LeakyReLU(alpha=0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        
        
        # Output layers
        pitch_output = Dense(128, activation='softmax',kernel_initializer='he_normal', name='pitch')(x)
        step_output = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='step')(x)
        duration_output = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='duration')(x)
    
        
        # Build the model
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        outputs = {
            'pitch': pitch_output,
            'step': step_output,
            'duration': duration_output,
        }

        model = Model(inputs, outputs)
        model.summary()

        model.compile(optimizer=self.generator_optimizer)
        return model
        
    def build_discriminator(self):
        inputs = Input(shape=(self.discriminator_output_dim[0], 3))  # 3 channels for pitch, step, and duration
        x = LSTM(128, return_sequences=True)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = LSTM(32)(x)
        x = Dense(1, kernel_initializer='he_normal')(x)
        model = Model(inputs, x)
        model.summary()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        model.compile(optimizer=self.critic_optimizer)
        return model

    def generate_data(self, num_samples):
        fake_data = self.generator(self.generate_noise(num_samples), training=False)
        # Compute the weighted sum of the pitch logits to get a differentiable approximation of the pitch
        pitch_range = tf.range(128, dtype=tf.float32)  # Assuming 128 possible pitch values
        fake_data_notes = tf.reduce_sum(fake_data['pitch'] * pitch_range, axis=-1, keepdims=True)
        # ensure fake_data_notes is int
        fake_data_notes = tf.cast(fake_data_notes, tf.int32)
        fake_data_notes = tf.cast(fake_data_notes, tf.float32)
        # Concatenate pitch, step, and duration for the critic input
        fake_data = tf.concat([fake_data_notes, fake_data['step'], fake_data['duration']], axis=-1)
        
        return fake_data

        
    def generate_real_batch(self, real_data, batch_size):
        batched_data = [real_data[i:i + batch_size] for i in range(0, len(real_data), batch_size)]
    # # Ensure each batch is correctly shaped
        batched_data = [batch.reshape(-1, batch_size, 1) for batch in batched_data if batch.shape[0] == batch_size]
        return  batched_data
    
    def generate_noise(self, batch_size):
        return tf.random.normal([batch_size, self.input_dim[0], self.input_dim[1]])
    
    @tf.function
    def lipschitz_penalty(self, real_data):
        with tf.GradientTape() as tape:
            tape.watch(real_data)
            real_scores = self.critic(real_data, training=True)
        
        gradients = tape.gradient(real_scores, [real_data])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        lipschitz_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        return lipschitz_penalty

    @tf.function
    def gradient_penalty(self, real_data, fake_data, batch_size):
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = epsilon * tf.cast(real_data, tf.float32) + (1 - epsilon) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interpolated_scores = self.critic(interpolated, training=True)

        gradients = tape.gradient(interpolated_scores, interpolated)
        gradients = tf.clip_by_value(gradients, -1.0, 1.0)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))

        return gradient_penalty
    @tf.function
    def train_critic_realData(self, real_data):
        lambda_lp = tf.constant(15.0, dtype=tf.float32)
        mcritic_loss = []
        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(real_data)
            real_scores_penalty = self.critic(real_data, training=True)
        with tf.GradientTape() as tape:
            real_scores = self.critic(real_data, training=True)
            critic_loss = tf.reduce_mean(real_scores)

            gradients = penalty_tape.gradient(real_scores_penalty, real_data)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            lipschitz_penalty = tf.reduce_mean(tf.square(tf.clip_by_value(grad_norm - 1.0, 0.0, np.infty)))
            # convert lipschitz_penalty to float32
            lipschitz_penalty = tf.cast(lipschitz_penalty, tf.float32)
            total_loss = critic_loss + lambda_lp * lipschitz_penalty


            # Combine critic loss with the Lipschitz penalty

        # Compute gradients of the total loss with respect to the critic's trainable variables
        gradients = tape.gradient(total_loss, self.critic.trainable_variables)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # Example of tighter clipping


        # Apply the gradients to update the critic's weights
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Accumulate the loss
        mcritic_loss.append(total_loss)

        return mcritic_loss
    @tf.function
    def train_critic_interpolated(self, real_data, batch_size):
        lambda_lp = tf.constant(15.0, dtype=tf.float32)
        mcritic_loss = []

        # Generate fake data from the generator
        fake_data = self.generator(self.generate_noise(batch_size), training=False)
        pitch_range = tf.range(128, dtype=tf.float32)
        fake_data_notes = tf.reduce_sum(fake_data['pitch'] * pitch_range, axis=-1, keepdims=True)
        fake_data_notes = fake_data_notes / 127.0
        fake_data = tf.concat([fake_data_notes, fake_data['step'], fake_data['duration']], axis=-1)

        # Interpolate between real and fake data
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated_data = epsilon * tf.cast(real_data, tf.float32) + (1 - epsilon) * fake_data
        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(interpolated_data)
            interpolated_scores_penalty = self.critic(interpolated_data, training=True)
            
        
        with tf.GradientTape() as tape:
            real_scores_interpolated = self.critic(interpolated_data, training=True)

            critic_loss = -tf.reduce_mean(real_scores_interpolated)

            gradients = penalty_tape.gradient(interpolated_scores_penalty, interpolated_data)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            lipschitz_penalty = tf.reduce_mean(tf.square(tf.clip_by_value(grad_norm - 1.0, 0.0, np.infty)))
            # convert lipschitz_penalty to float32
            lipschitz_penalty = tf.cast(lipschitz_penalty, tf.float32)
            total_loss = critic_loss + lambda_lp * lipschitz_penalty

        # The total loss for the critic on the interpolated data

        # Compute gradients of the total loss with respect to the critic's trainable variables
        gradients = tape.gradient(total_loss, self.critic.trainable_variables)
        
        # Clip gradients to the specified range
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]

        # Apply the gradients to update the critic's weights
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Accumulate the loss
        mcritic_loss.append(total_loss)

        return mcritic_loss


    @tf.function
    def train_critic_generatedData(self, real_data, batch_size):
        lambda_lp = tf.constant(15.0, dtype=tf.float32)
        
        # Generate fake data
        data = self.generator(self.generate_noise(batch_size), training=False)
        pitch_range = tf.range(128, dtype=tf.float32)
        fake_data_notes = tf.reduce_sum(data['pitch'] * pitch_range, axis=-1, keepdims=True)
        fake_data_notes = fake_data_notes / 127.0
        fake_data = tf.concat([fake_data_notes, data['step'], data['duration']], axis=-1)
        
        # Interpolate between real and fake data
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated_data = epsilon * tf.cast(real_data, tf.float32) + (1 - epsilon) * fake_data

        with tf.GradientTape() as tape:
            # Calculate the critic's scores
            real_scores = self.critic(real_data, training=True)
            fake_scores = self.critic(fake_data, training=True)
            critic_loss =  tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        
        # Compute gradient penalty
            with tf.GradientTape() as penalty_tape:
                penalty_tape.watch(interpolated_data)
                interpolated_scores = self.critic(interpolated_data, training=True)
            
            gradients = penalty_tape.gradient(interpolated_scores, interpolated_data)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            lipschitz_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
            
            # Combine the loss with the gradient penalty
            total_loss = critic_loss + lambda_lp * lipschitz_penalty

        # Apply gradients to the critic's weights
        gradients = tape.gradient(total_loss, self.critic.trainable_variables)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        return total_loss
    # should be uncommented
    @tf.function
    def generator_train_step(self, batch_size):      
        with tf.GradientTape() as tape:
            # Generate fake data from the generator
            fake_data = self.generator(self.generate_noise(batch_size), training=True)
            
            # # Compute the weighted sum of the pitch logits to get a differentiable approximation of the pitch
            pitch_range = tf.range(128, dtype=tf.float32)  # Assuming 128 possible pitch values
            fake_data_notes = tf.reduce_sum(fake_data['pitch'] * pitch_range, axis=-1, keepdims=True)
            
            # Normalize the pitch values to [0, 1] range
            fake_data_notes = fake_data_notes / 127.0  # Since pitch values range from 0 to 127
            
            # Concatenate pitch, step, and duration for the critic input
            fake_data = tf.concat([fake_data_notes, fake_data['step'], fake_data['duration']], axis=-1)
            
            # Compute the critic's output for the generated data
            generated_scores = self.critic(fake_data, training=False)
            
            # The generator aims to minimize the negative of the critic's output (Wasserstein loss)
            gen_loss = -tf.reduce_mean(generated_scores)
        
        # Compute the gradients of the generator loss with respect to the generator's trainable variables
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        
        # Apply the gradients to update the generator's weights
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return gen_loss

    @tf.function
    def train_step(self, real_data, batch_size=4):
        # crit_losses = self.train_critic_realData(real_data)
        crit_losses = self.train_critic_generatedData(real_data, batch_size)
        # crit_losses.extend(self.train_critic_interpolated(real_data, batch_size))
        gen_loss = self.generator_train_step(batch_size)
        return gen_loss, crit_losses
    
    # @tf.function(jit_compile=True)
    def train_epoch(self, dataset: tf.data.Dataset, batch_size=4):
        epoch_gen_loss = []
        epoch_disc_loss = []
        print("Training...")
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        for real_batch in tqdm(dataset):
            gen_loss, disc_loss = self.train_step(real_batch, batch_size)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss) 
            with self.train_summary_writer.as_default(): 
                    tf.summary.scalar('Discriminator Loss', disc_loss, step=self.train_step_counter)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=self.train_step_counter)            
            self.train_step_counter.assign_add(1)
        return epoch_gen_loss, epoch_disc_loss

    def train(self, dataset, num_epochs, patience=10, batch_size=4):
        if self.load:
            print("Model loaded, skipping training")
            return
        logdir = "src/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(logdir)
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)
        best_loss = float('inf')
        patience_counter = 0
        generator_generated_data = []
        # profiler.start(logdir)
        actual_epochs = 0
        for epoch in range(num_epochs):
            epoch_gen_loss, epoch_disc_loss = self.train_epoch(dataset, batch_size)
            self.train_summary_writer.flush()

            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            print(f"Epoch {epoch}: Generator Loss = {avg_gen_loss}, Discriminator Loss = {avg_disc_loss}")
            avg_disc_loss = abs(avg_disc_loss)
            # Ignore the first half of the epochs for early stopping
            if avg_disc_loss < best_loss:
                best_loss = avg_disc_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter} / {patience}")
                if patience_counter >= patience:
                    print("Early stopping...")
                    break
        self.generator.save("generator.keras")
        self.critic.save("discriminator.keras")
        return generator_generated_data
        # profiler.stop() 
    # @tf.function(jit_compile=True)
    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def discriminator_loss(self, real_predictions, generated_predictions, real_labels, generated_labels):
        real_loss = tf.keras.losses.BinaryCrossentropy()(real_labels, real_predictions)
        generated_loss = tf.keras.losses.BinaryCrossentropy()(generated_labels, generated_predictions)
        total_loss = real_loss + generated_loss
        return total_loss
    
    def generator_loss(self, generated_predictions, real_labels):
        return tf.keras.losses.BinaryCrossentropy()(real_labels, generated_predictions)
    
    def save(self, path):
        self.generator.save(path)
        print(f"Model saved to {path}")
    def load(self, path):
        self.generator = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
