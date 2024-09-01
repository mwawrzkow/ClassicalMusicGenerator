import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Bidirectional, BatchNormalization, Reshape, Input, Flatten, LayerNormalization, Dropout
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
tf.config.optimizer.set_jit(True)
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# tf.config.optimizer.set_jit('autoclustering')
logdir = None
tensorboard_callback = None
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
        
        # Output layers
        pitch_output = Dense(128, activation='softmax',kernel_initializer='he_normal', name='pitch')(x)
        step_output = Dense(1, kernel_initializer='he_normal',name='step')(x)
        duration_output = Dense(1, kernel_initializer='he_normal', name='duration')(x)
        
        # Build the model
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
        outputs = {
            'pitch': pitch_output,
            'step': step_output,
            'duration': duration_output,
        }

        model = Model(inputs, outputs)
        loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        'step': tf.keras.losses.MeanSquaredError(),
        'duration': tf.keras.losses.MeanSquaredError(),
    }
        model.summary()

        model.compile(
            optimizer=self.generator_optimizer,
            loss=loss,
            loss_weights={
                'pitch': 1.0,
                'step': 1.0,
                'duration': 1.0,
            })
        return model
        
    def build_discriminator(self):
        model = Sequential()
        model.add(Input(shape=(self.discriminator_output_dim[0], 3)))  # 3 channels for pitch, step, and duration
        model.add(LSTM(128))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear', kernel_initializer='he_normal'))  # Binary classification output
        model.summary()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)
        model.compile(optimizer=self.critic_optimizer)
        return model
        
    def normalize_data(self, data, min_val=-1, max_val=1):
        """Normalize data to be between min_val and max_val."""
        data_min = tf.reduce_min(data)
        data_max = tf.reduce_max(data)
        return min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)
    
    def generate_increasing_start_times(self, batch_size):
        """Generate start times in increasing order between -1 and 1."""
        start_times = np.sort(np.random.uniform(-1, 1, batch_size))
        return tf.convert_to_tensor(start_times, dtype=tf.float32)

    def generate_data(self, num_samples):
        noise = self.generate_noise(num_samples)
        fake_data = self.generator(noise, training=False)
        
        # Get the index of the maximum logit (the most probable pitch)
        pitch = tf.argmax(fake_data['pitch'], axis=-1)
        
        # Convert pitch to the desired range [0, 127]
        pitch = tf.clip_by_value(tf.cast(pitch, tf.int32), 0, 127)
        
        # Ensure step and duration are non-negative
        step = tf.maximum(0.0, tf.squeeze(fake_data['step'], axis=-1))
        duration = tf.maximum(0.0, tf.squeeze(fake_data['duration'], axis=-1))
        
        # Convert pitch to float32 to match step and duration data types
        pitch = tf.cast(pitch, tf.float32)
        
        # Combine outputs into a single array with shape (num_samples, sequence_length, 3)
        generated_data = tf.stack([pitch, step, duration], axis=-1)
        
        return generated_data

        
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
        epsilon = tf.random.uniform(shape=[batch_size, 1,], minval=0.0, maxval=1)
        interpolated = epsilon * tf.cast(real_data, tf.float32) + ((1 - epsilon) * fake_data)
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_scores = self.critic(interpolated, training=True)
        
        gradients = gp_tape.gradient(interpolated_scores, [interpolated])[0]
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        return gradient_penalty


    @tf.function
    def train_critic_Interpolated(self, real_data, batch_size): 
        lambda_lp = 4.0
        mcritic_loss = []
        
        # for _ in range(5):
        noise = self.generate_noise(batch_size)
        fake_data = self.generator(noise, training=False)
        
        # Generate interpolated data
        epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        
        with tf.GradientTape() as tape:
            # real_scores = self.critic(real_data, training=True)
            # fake_scores = self.critic(fake_data, training=True)
            
            # Compute the Lipschitz penalty for interpolated data
            lipschitz_penalty_value = self.lipschitz_penalty(interpolated)
            
            # Critic loss
            critic_loss = (-tf.reduce_mean(interpolated) * epsilon) + lambda_lp * lipschitz_penalty_value

            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]  # Gradient clipping
            mcritic_loss.append(critic_loss)
            self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        return mcritic_loss
    @tf.function
    def calculate_penalty(self, data, ndims = 2):
        # Assuming `data` is a tuple where the first element is the tensor you need
        tensor_data = data[0]  # Extract the tensor part of the tuple
        duration = []
        if ndims == 3:
            duration = tensor_data[:, :, 2]  # Correctly index the duration field
        else:
            duration = tensor_data[:, 2]
        long_note_threshold = 1.0  # Define a threshold for long notes
        long_notes = tf.greater(duration, long_note_threshold)
        penalty_factor = tf.where(long_notes, 2.0, 1.0)  # Apply higher penalty to long notes
        penalty_factor = tf.reshape(penalty_factor, [-1, 1])  # Reshape to be broadcastable with grad_norm
        return penalty_factor

    # should be uncommented 
    @tf.function 
    def train_critic_realData(self, real_data):
        lambda_lp = 5.0
        mcritic_loss = []

        penalty_factor = self.calculate_penalty(real_data)


        with tf.GradientTape() as tape:
            tape.watch(real_data[0])
            real_scores = self.critic(real_data[0], training=True)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(real_scores), axis=[1]))
            lipschitz_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0) + tf.square(grad_norm - 1.0) * penalty_factor)
            critic_loss = -tf.reduce_mean(real_scores) + lambda_lp * lipschitz_penalty  # Compute the critic loss

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.critic_optimizer.apply_gradients(zip(clipped_gradients, self.critic.trainable_variables))
        mcritic_loss.append(critic_loss)

        return mcritic_loss

    @tf.function
    def train_critic_generatedData(self, batch_size):
        lambda_lp = 6.0
        mcritic_loss = []
        # for _ in range(3):
        # noise = self.generate_noise(batch_size)
        fake_data = self.generator(self.generate_noise(batch_size), training=False)
        # from generated batch get only notes that are the most probable
        fake_data_notes = tf.argmax(fake_data['pitch'], axis=-1)
        fake_data_notes = tf.clip_by_value(tf.cast(fake_data_notes, tf.int32), 0, 127)
        fake_data_notes = tf.cast(fake_data_notes, tf.float32)
        fake_data_notes = tf.expand_dims(fake_data_notes, axis=-1)  # Shape [512] -> [512, 1]

        fake_data = tf.concat([fake_data_notes, fake_data['step'], fake_data['duration']], axis=-1)
        # tf.print(tf.shape(fake_data))
        penalty_factor = self.calculate_penalty(fake_data,2)

        with tf.GradientTape() as tape:
            tape.watch(fake_data)
            real_scores = self.critic(fake_data, training=True)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(real_scores), axis=[1]))
            lipschitz_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0)* penalty_factor)
            critic_loss = -tf.reduce_mean(real_scores) + lambda_lp * lipschitz_penalty  # Compute the critic loss

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Filter out None gradients and apply tf.clip_by_value only to valid gradients
        clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        # valid_gradients = [(grad, var) for grad, var in zip(clipped_gradients, self.critic.trainable_variables) if grad is not None]
        # clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.critic_optimizer.apply_gradients(zip(clipped_gradients, self.critic.trainable_variables))
        mcritic_loss.append(critic_loss)
        return mcritic_loss

    # should be uncommented
    @tf.function
    def generator_train_step(self, batch_size):      
        with tf.GradientTape() as tape:
            fake_data = self.generator(self.generate_noise(batch_size), training=True)
            pitch_range = tf.range(128, dtype=tf.float32)
            fake_data_notes = tf.reduce_sum(fake_data['pitch'] * pitch_range, axis=-1, keepdims=True)
            
            # Concatenate with step and duration
            fake_data = tf.concat([fake_data_notes, fake_data['step'], fake_data['duration']], axis=-1)
            generated_scores = self.critic(fake_data, training=False)
            gen_loss = -tf.reduce_mean(generated_scores)
        # gen_gradients = self.generator_loss(generated_scores, tf.ones_like(generated_scores))
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        # gen_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gen_gradients]
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        return gen_loss
    # @tf.function
    def train_step(self, real_data):
        batch_size = 512
        crit_losses = []
        # Train the critic with real data
        for _ in range(5):
            real_critic_loss = self.train_critic_realData(real_data)
            
            # Train the critic with generated data
            generated_critic_loss = self.train_critic_generatedData(batch_size)
            crit_losses.extend(real_critic_loss)
            crit_losses.extend(generated_critic_loss)
        
        # Train the critic with interpolated data
        # interpolated_critic_loss = self.train_critic_Interpolated(real_data, batch_size)

        # Train the generator
        gen_loss = self.generator_train_step(batch_size)
        
        return gen_loss, crit_losses#+ interpolated_critic_loss
    # @tf.function(jit_compile=True)
    def train_epoch(self, dataset: tf.data.Dataset):
        epoch_gen_loss = []
        epoch_disc_loss = []
        print("Training...")
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        for real_batch in tqdm(dataset):
            gen_loss, disc_loss = self.train_step(real_batch)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.extend(disc_loss)
            for loss in disc_loss: 
                with self.train_summary_writer.as_default(): 
                    tf.summary.scalar('Discriminator Loss', loss, step=self.train_step_counter)
                self.train_step_counter.assign_add(1)        
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=self.train_step_counter)
                self.train_summary_writer.flush()
            
            self.train_step_counter.assign_add(1)
        return epoch_gen_loss, epoch_disc_loss, self.generator(self.generate_noise(512), training=False)

    def train(self, dataset, num_epochs, patience=10):
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
            epoch_gen_loss, epoch_disc_loss, generated_data = self.train_epoch(dataset)
            generator_generated_data.append(generated_data)
            
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
