from copy import deepcopy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, GRU, LeakyReLU, Bidirectional, Conv1D, Lambda, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow_addons.layers import SpectralNormalization



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
class PitchMaskLayer(layers.Layer):
    def __init__(self, min_max, **kwargs):
        super(PitchMaskLayer, self).__init__(**kwargs)
        self.min_max = min_max
    
    def call(self, pitch_probs):
        pitch_min_idx = self.min_max['pitch_min']
        pitch_max_idx = self.min_max['pitch_max']

        mask = tf.concat([
            tf.fill([pitch_min_idx], 1e-9),
            tf.fill([pitch_max_idx - pitch_min_idx + 1], 1.0),
            tf.fill([128 - pitch_max_idx - 1], 1e-9)
        ], axis=0)

        mask = tf.reshape(mask, (1, 1, 128))
        pitch_probs = pitch_probs * mask
        pitch_probs = pitch_probs / tf.reduce_sum(pitch_probs, axis=-1, keepdims=True)
        return pitch_probs
def apply_pitch_mask(pitch_probs, min_max):
    # Create a mask where valid pitches are 1 and invalid pitches are a small number (like 1e-9)
    pitch_min_idx = min_max['pitch_min']  # Example: 22
    pitch_max_idx = min_max['pitch_max']  # Example: 105

    # Create a mask of shape [128] that is 1 for valid notes and 1e-9 for invalid notes
    mask = tf.concat([
        tf.fill([pitch_min_idx], 1e-9),  # For notes below pitch_min
        tf.fill([pitch_max_idx - pitch_min_idx + 1], 1.0),  # For valid notes in the range
        tf.fill([128 - pitch_max_idx - 1], 1e-9)  # For notes above pitch_max
    ], axis=0)

    # Reshape mask to broadcast it over batch and sequence dimensions
    mask = tf.reshape(mask, (1, 1, 128))  # Shape: [1, 1, 128] to match pitch_output shape

    # Apply the mask to pitch probabilities
    pitch_probs = pitch_probs * mask

    # Re-normalize the probabilities so they sum to 1 (apply softmax again)
    pitch_probs = pitch_probs / tf.reduce_sum(pitch_probs, axis=-1, keepdims=True)

    return pitch_probs
class GAN(BaseModel):
    """
    Generative Adversarial Network (GAN) class.
    """

    def __init__(self, input_dim, generator_output_dim, discriminator_output_dim, load=False, min_max= None, continue_training=False):
        """
        Constructor for the GAN class.

        Parameters:
        - input_dim (int): Dimension of the input noise vector.
        - generator_output_dim (int): Dimension of the generator output.
        - discriminator_output_dim (int): Dimension of the discriminator output.
        """
        # print available devices
        print(f"Available devices: {tf.config.list_physical_devices('GPU')}")
        self.generator = None
        self.critic = None
        # if no GPU is available, crash
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError("No GPU found, can't continue.")
        if min_max is not None:
            print("Min max values found")
            print("The generator will be fine tuned to generate values supported by the dataset")
            self.min_max = min_max
        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim
        self.discriminator_output_dim = discriminator_output_dim
        self.continue_training = continue_training
        # if load is True, load the model
        self.load = load
        if load:
            if self.generator is None:
                print("Model not initialized, initializing model")
                self.generator = self.build_generator()
            self.generator.load_weights("generator.keras")
            if self.critic is None:
                print("Model not initialized, initializing model")
                self.critic = self.build_discriminator()
            self.critic.load_weights("discriminator.keras")
            print("Model loaded successfully.")

            

        else: 
            self.generator = self.build_generator()
            self.critic = self.build_discriminator()
            self.generator.save_weights("generator.keras")


        
    def build_generator(self):
        inputs = Input(shape=(self.input_dim))

        # LSTM layer to process the input
        x = LSTM(256, return_sequences=True)(inputs)

        # Spectral Normalization applied to the Dense layers for each output

        # Pitch output (fine-tuned range: [pitch_min, pitch_max])
        pitch_output = SpectralNormalization(Dense(128, activation='softmax', kernel_initializer='he_normal', name='pitch_output'))(x)
        # min_max = deepcopy(self.min_max)
        # Apply the pitch mask
        pitch_output = SpectralNormalization(Dense(128, activation='softmax', kernel_initializer='he_normal', name='pitch_output'))(x)

        duration_output = SpectralNormalization(Dense(1, activation="sigmoid", kernel_initializer='he_normal', name='duration_output'))(x)
        duration_output = Lambda(lambda x: self.min_max['duration_min'] + x * (self.min_max['duration_max'] - self.min_max['duration_min']))(duration_output)

        # Step output (fine-tuned range: [step_min, step_max])
        step_output = SpectralNormalization(Dense(1, kernel_initializer='he_normal', name='step_output'))(x)
        step_output = Lambda(lambda x: tf.nn.relu(x) * self.min_max['step_max'])(step_output)  # Ensure step is non-negative and scaled

        # Velocity output (fine-tuned range: [velocity_min, velocity_max])
        velocity_output = SpectralNormalization(Dense(1, activation="sigmoid", kernel_initializer='he_normal', name='velocity_output'))(x)
        velocity_output = Lambda(lambda x: self.min_max['velocity_min'] + x * (self.min_max['velocity_max'] - self.min_max['velocity_min']))(velocity_output)


        # Build the model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.96, staircase=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.0, beta_2=0.9)

        outputs = {
            'pitch': pitch_output,
            'duration': duration_output,
            'step': step_output,
            'velocity': velocity_output
        }

        model = Model(inputs, outputs)
        model.summary()

        model.compile(optimizer=self.generator_optimizer)
        if self.continue_training:
            model.load_weights("generator.keras")
        return model
        
    def build_discriminator(self):
        note_input = Input(shape=(self.discriminator_output_dim[0], 128), name='pitch')
        step_input = Input(shape=(self.discriminator_output_dim[0], 1), name='step')
        duration_input = Input(shape=(self.discriminator_output_dim[0], 1), name='duration')
        velocity_input = Input(shape=(self.discriminator_output_dim[0], 1), name='velocity')
        x = layers.concatenate([note_input, step_input, duration_input, velocity_input], axis=-1)

        # Process both inputs with LSTMs or any other layers
        x = LSTM(256, return_sequences=True)(x)
        x = SpectralNormalization(Dense(1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)  # Output layer


        # Build the model with two inputs
        model = Model(inputs=[note_input, step_input, duration_input, velocity_input], outputs=x)
        model.summary()
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.000007)
        model.compile(optimizer=self.critic_optimizer)
        if self.continue_training:
            model.load_weights("discriminator.keras")
        return model

    def generate_data(self, num_samples, temperature=1.0):
        # Step 1: Generate fake pitch and step data from the generator
        fake_data = self.generator(self.generate_noise(num_samples), training=False)

        # Step 2: Adjust softmax probabilities using temperature
        fake_data_probs = fake_data['pitch']  # Shape: [batch_size, seq_length, 128]

        # Adjust the logits with temperature for softmax-like sampling
        fake_data_probs = tf.math.log(fake_data_probs + 1e-10) / temperature
        fake_data_probs = tf.nn.softmax(fake_data_probs, axis=-1)

        # Step 3: Sample the pitch from the softmax-like probabilities
        fake_data_notes = tf.random.categorical(tf.reshape(fake_data_probs, [-1, 128]), 1)
        fake_data_notes = tf.reshape(fake_data_notes, [num_samples, -1, 1])  # Reshape back to original shape

        # Convert fake_data_notes to float32 to match the type of step_output and fake_data_duration
        fake_data_notes = tf.cast(fake_data_notes, tf.float32)

        # Step 4: Get the generated step data
        fake_data_duration = fake_data['duration']  # Shape: [batch_size, seq_length, 1]
        fake_data_velocity = fake_data['velocity']  # Shape: [batch_size, seq_length, 1]

        # Step 5: One-hot encode the generated pitch for RNN input
        fake_data_one_hot = tf.one_hot(tf.cast(fake_data_notes, tf.int32), depth=128, axis=-1)  # Shape: [batch_size, seq_length, 1, 128]

        # Step 6: Remove the extra dimension to get the correct input shape for the predictor
        fake_data_one_hot = tf.squeeze(fake_data_one_hot, axis=-2)  # Shape: [batch_size, seq_length, 128]

        # Step 7: Use the RNN predictor to predict the duration based on the generated pitch and step
        # step_output = self.predictor([fake_data_one_hot, fake_data_duration], training=False)  # Pass both pitch and step
        fake_data_step = fake_data['step']
        # Step 8: Concatenate pitch, step, and duration to form the final generated data
        generated_data = tf.concat([fake_data_notes, fake_data_step, fake_data_duration, fake_data_velocity], axis=-1)  # Shape: [batch_size, seq_length, 3]

        return generated_data  # Return the full generated sequence with pitch, step, and duration

    def generate_real_batch(self, real_data, batch_size):
        batched_data = [real_data[i:i + batch_size] for i in range(0, len(real_data), batch_size)]
    # # Ensure each batch is correctly shaped
        batched_data = [batch.reshape(-1, batch_size, 1) for batch in batched_data if batch.shape[0] == batch_size]
        return  batched_data
    
    def generate_noise(self, batch_size):
        # Generate noise with normal distribution
        normal_noise = tf.random.normal([batch_size, self.input_dim[0], self.input_dim[1]])
        
        # Optionally, add uniform noise or any other type of randomness for more diversity
        uniform_noise = tf.random.uniform([batch_size, self.input_dim[0], self.input_dim[1]], minval=-1, maxval=1)
        
        # Combine both noise vectors (optional, depending on the architecture)
        combined_noise = normal_noise + uniform_noise
        
        return combined_noise

    @tf.function
    def gradient_penalty(self, real_data, fake_data):
        """
        Compute the gradient penalty for WGAN-GP.
        Inputs:
            real_data: Real samples (batch)
            fake_data: Fake samples (batch)
        """
        if real_data[0].shape[0] != fake_data[0].shape[0]:
            raise ValueError("Real data and fake data must have the same batch size")

        # Interpolation factor epsilon
        epsilon = tf.random.uniform([real_data[0].shape[0]] + [1 for _ in range(len(real_data[0].shape) - 1)], 0.0, 1.0)

        # Interpolated data between real and fake
        interpolated_data_pitch = epsilon * real_data[0] + (1 - epsilon) * fake_data[0]
        interpolated_data_step = epsilon * real_data[1] + (1 - epsilon) * fake_data[1]
        interpolated_data_duration = epsilon * real_data[2] + (1 - epsilon) * fake_data[2]
        interpolated_data_velocity = epsilon * real_data[3] + (1 - epsilon) * fake_data[3]
        interpolated_data = [interpolated_data_pitch, interpolated_data_step, interpolated_data_duration, interpolated_data_velocity]

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)
            interpolated_scores = self.critic(interpolated_data, training=True)

        gradients = tape.gradient(interpolated_scores, interpolated_data)

        # Compute L2 norm of the gradients
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), axis=[1, 2]) + tf.reduce_sum(tf.square(gradients[1]), axis=[1, 2]))
        
        # Gradient penalty as per WGAN-GP
        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        return gradient_penalty

    @tf.function
    def compute_d_loss(self, real_data, batch_size):
        """
        Compute the discriminator loss with gradient penalty.
        Inputs:
            real_data: Real samples from the dataset
            batch_size: Size of the batch
        """
        # Generate fake data from the generator
        fake_data = self.generator(self.generate_noise(batch_size), training=False)
        real_data_pitch = real_data[:, :, :128]  # Real pitch (first 128 dimensions)
        real_data_step = real_data[:, :, 128:129]  # Real step (next dimension)
        real_data_duration = real_data[:, :, 129:130]  # Real duration (last dimension)
        real_data_velocity = real_data[:, :, 130:131]  # Real velocity (last dimension)

        real_data = [real_data_pitch, real_data_step, real_data_duration, real_data_velocity] 
        # Get discriminator scores for real and fake data
        real_scores = self.critic(real_data, training=True)
        fake_scores = self.critic(fake_data, training=True)

        # Compute the WGAN loss
        d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

        # Compute the gradient penalty
        # gp = self.gradient_penalty(real_data, fake_data)

        # Total discriminator loss with gradient penalty
        total_d_loss = d_loss #+ 10.0 * gp  # λ = 10 for gradient penalty as per the original WGAN-GP paper

        return total_d_loss
    
    @tf.function
    def train_d_step(self, real_data, batch_size, grad_norm_th=7.0):
        """
        Training step for the discriminator with gradient penalty and gradient clipping.
        Inputs:
            real_data: Real samples from the dataset
            batch_size: Batch size for training
            grad_norm_th: Threshold for gradient clipping
        """
        with tf.GradientTape() as disc_tape:
            disc_loss = self.compute_d_loss(real_data, batch_size)

        # Compute the gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.critic.trainable_variables)

        # Clip the gradients by global norm
        gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, grad_norm_th)

        # Apply the gradients to update the critic/discriminator
        self.critic_optimizer.apply_gradients(zip(gradients_of_discriminator, self.critic.trainable_variables))

        # Optionally, compute and return the gradient norm for logging
        # gradient_norm = tf.linalg.global_norm(gradients_of_discriminator)
        
        return disc_loss#, gradient_norm
    
    @tf.function
    def train_generator_real(self, batch_size, real_data): 
            with tf.GradientTape() as tape:
                generated_scores = self.critic(self.generator(self.generate_noise(batch_size), training=True), training=False)
                generator_loss = -tf.reduce_mean(generated_scores)
            

            gen_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables)) 
            return generator_loss
        # Return the generator loss for logging/monitoring 

    def train_step(self, real_data, batch_size=4, is_generator_step=False):
        crit_losses = self.train_d_step(real_data, batch_size)
        gen_loss = None
        if is_generator_step:
            gen_loss = self.train_generator_real(batch_size, real_data)
        return gen_loss, crit_losses
    
    def train_epoch(self, dataset: tf.data.Dataset, batch_size=4, critic_giga_steps=5):
        epoch_gen_loss = []
        epoch_disc_loss = []
        step = 0
        print() # alocate a new line
        for real_batch in dataset:
            is_generator_step = step % critic_giga_steps == 0
            gen_loss, disc_loss = self.train_step(real_batch, batch_size, is_generator_step)
            if gen_loss is not None:
                epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss) 
            step += 1
            print(f"\rStep {step}: Generator Loss = {np.mean(epoch_gen_loss) }, Discriminator Loss = {np.mean(epoch_disc_loss)}", end="")
        print("\r", end="")
        return epoch_gen_loss, epoch_disc_loss, step
    def train(self, dataset, num_epochs, patience=10, batch_size=4):
        if self.load:
            print("Model loaded, skipping training")
            return
        logdir = "src/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(logdir)
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)
        patience_counter = 0
        generator_generated_data = []
        # profiler.start(logdir)
        critic_giga_steps = 3
        idx = 0 
        for epoch in range(num_epochs):
            start_time = datetime.now()
            epoch_gen_loss, epoch_disc_loss, steps = self.train_epoch(dataset, batch_size,critic_giga_steps)
            end_time = datetime.now()
            self.train_summary_writer.flush()
            print(f"Epoch {epoch} took {end_time - start_time}, possible train remaining time: {(end_time - start_time) * (num_epochs - epoch)}")
            # check if critic or generator colapsed 
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            # loop through generator loss and discriminator loss and save losses per step 
            
            with self.train_summary_writer.as_default(): 
                tf.summary.scalar('Critic Loss epoch', avg_disc_loss, step=self.train_step_counter)
                tf.summary.scalar('Generator Loss epoch', avg_gen_loss, step=self.train_step_counter)                   
            self.train_step_counter.assign_add(1)
            print(f"Epoch {epoch}: Generator Loss = {avg_gen_loss}, Discriminator Loss = {avg_disc_loss}, Steps = {steps}")
            # Ignore the first half of the epochs for early stopping
            critic_giga_steps = 3 if avg_disc_loss < -2.0 else 1
            if avg_disc_loss < -2.0:
                print("Error in training, has become too sure of itself. Model propably collapsed, thus incrementing patience")
                patience_counter += 1
            else: 
                patience_counter = 0
                
            avg_disc_loss = abs(avg_disc_loss)
            
            if patience_counter >= patience:
                print("Early stopping")
                break
            if idx % 10 == 0:
                self.generator.save_weights("generator.keras")
                self.critic.save_weights("discriminator.keras")
  
 
        self.generator.save_weights("generator.keras")
        self.critic.save_weights("discriminator.keras")
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
        self.generator.save_weights(path)
        print(f"Model saved to {path}")
    def load(self, path):
        generator = path + "/generator.keras"
        # check if generator exists
        if self.generator is None:
            print(f"Model not initialized, initializing model")
            self.critic = self.build_generator()
        self.generator.load_weights(generator)
        if self.critic is None:
            print(f"Model not initialized, initializing model")
            self.critic = self.build_discriminator()
        critic = path + "/discriminator.keras"
        self.critic.load_weights(critic)

        print(f"Model loaded from {path}")
