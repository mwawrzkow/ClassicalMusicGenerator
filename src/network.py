import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Bidirectional, BatchNormalization, Reshape, Input, Flatten
# test also with tensorflow.python.keras 
from tqdm import tqdm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.profiler import profiler_v2 as profiler
from datetime import datetime
from tensorflow.python.client import device_lib
# tf.debugging.set_log_device_placement(True)
print(device_lib.list_local_devices())
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("No GPU found, can't continue.")
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# tf.config.optimizer.set_jit('autoclustering')
logdir = None
tensorboard_callback = None
class GAN:
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
            # should we compile the model again?

        else: 
            self.generator = self.build_generator()
            self.critic = self.build_discriminator()
            
            # self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            self.generator.compile(optimizer=self.generator_optimizer)
            self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            self.critic.compile(optimizer=self.critic_optimizer)
        
    def build_generator(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim)))  # Adding sequence length of 1
        model.add(Dense(128)) # for now simplify 
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(128))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        # # model.add(Bidirectional(LSTM(128, return_sequences=False)))
        # model.add(Dense(256, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        num_output_units = np.prod(self.generator_output_dim) 
        model.add(Flatten())
        # model.add(Dense(num_output_units, activation='tanh'))    
        model.add(Dense(self.generator_output_dim[0], activation='tanh'))    
        # model.add(Reshape(self.generator_output_dim))    
        # model.add(Dense(self.generator_output_dim[0], activation='relu'))
        model.summary()
        # model.add(LSTM(32, return_sequences=True))  # Maintain 3D output for sequence processing
        # model.add(Bidirectional(LSTM(64)))  # Combines forward and backward LSTM
        # model.add(Dense(256, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.generator_output_dim), activation='tanh'))
        # model.add(Reshape(self.generator_output_dim))
        return model
    
    def build_discriminator(self):
        model = Sequential()
        model.add(Input(shape=self.discriminator_output_dim))
        # make 3D
        model.add(Reshape((self.discriminator_output_dim[0], 1)))
        model.add(Dense(128)) # for now simplify
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(128))
        # model.add(layers.BatchNormalization())
        # model.add(layers.LeakyReLU())
        # # model.add(Bidirectional(LSTM(128, return_sequences=False)))
        # model.add(Dense(256, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        num_output_units = np.prod(self.generator_output_dim) 
        model.add(Flatten())
        # model.add(Dense(num_output_units, activation='tanh'))    
        model.add(Dense(self.generator_output_dim[0], activation='tanh')) 
        # model.add(Dense(256, activation='relu'))
        # model.add(Bidirectional(LSTM(256)))
        # model.add(Dense(256, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.9))
        # # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='linear'))
        model.summary()
        # model.add(LSTM(512, return_sequences=True))
        # model.add(Bidirectional(LSTM(512)))
        # model.add(Dense(512, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(512, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        return model
    
    def generate_data(self, num_samples):
        # generate only data which is not detected as fake by the discriminator
        noise = self.generate_noise(num_samples)
        generated_data = self.generator.predict(noise)
        critic_scores = self.critic.predict(generated_data)
        avarage_score = np.mean(critic_scores)
        print(f"Average critic score: {avarage_score}")
        return generated_data
        
    def generate_real_batch(self, real_data, batch_size):
        batched_data = [real_data[i:i + batch_size] for i in range(0, len(real_data), batch_size)]
    # # Ensure each batch is correctly shaped
        batched_data = [batch.reshape(-1, batch_size, 1) for batch in batched_data if batch.shape[0] == batch_size]
        return  batched_data
    def generate_noise(self, batch_size):
        # Generating noise with the correct shape for the generator
        return tf.random.normal([batch_size, self.input_dim[0], self.input_dim[1]])
    @tf.function
    def train_step(self, real_data):
        batch_size = 256
        lambda_gp = 10.0  # Gradient penalty coefficient
        seed_part1 = 4
        seed_part2 = 2
        seed = [seed_part1, seed_part2]
        for _ in range(5):
            noise = self.generate_noise(batch_size)
            with tf.GradientTape() as tape:
                real_scores = self.critic(real_data, training=True)
                fake_data = self.generator(noise, training=False)
                fake_scores = self.critic(fake_data, training=True)

                # Example usage of stateless random function with a seed
                alpha = tf.random.stateless_uniform(shape=(batch_size, 1), seed=seed, minval=0.0, maxval=1.0)

                interpolated = real_data * alpha + fake_data * (1 - alpha)
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    interpolated_scores = self.critic(interpolated, training=True)
                gradients = gp_tape.gradient(interpolated_scores, [interpolated])[0]
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
                gradient_penalty = lambda_gp * tf.reduce_mean((grad_norm - 1.0) ** 2)

                critic_loss = (self.wasserstein_loss(-tf.ones_like(real_scores), real_scores) + 
                            self.wasserstein_loss(tf.ones_like(fake_scores), fake_scores) + 
                            gradient_penalty)

            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Generator update
        noise = self.generate_noise(batch_size)
        with tf.GradientTape() as tape:
            generated_data = self.generator(noise, training=True)
            generated_scores = self.critic(generated_data, training=True)
            gen_loss = -self.wasserstein_loss(tf.ones_like(generated_scores), generated_scores)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        return gen_loss, critic_loss



    # @tf.function(jit_compile=True)
    def train_epoch(self, dataset: tf.data.Dataset):
        epoch_gen_loss = []
        epoch_disc_loss = []
        # for idx, real_batch in enumerate(tqdm(dataset.prefetch(tf.data.experimental.AUTOTUNE))):
        #     # every 10 steps train both generator and discriminator
        #     # if idx % SKIP_GENERATOR == 0:
        #     gen_loss, disc_loss = self.train_step(real_batch)
        print("Training...")
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        for real_batch in tqdm(dataset):
            gen_loss, disc_loss = self.train_step(real_batch)
            # else:
            #     gen_loss, disc_loss = self.train_step(real_batch, train_only_discriminator=True)
            # Log every 1000 steps
            
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=self.train_step_counter)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=self.train_step_counter)
                self.train_summary_writer.flush()

            
            self.train_step_counter.assign_add(1)
        return epoch_gen_loss, epoch_disc_loss

    def train(self, dataset, num_epochs, patience=10):
        if self.load:
            print("Model loaded, skipping training")
            return
        logdir = "src/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)
        self.train_summary_writer = tf.summary.create_file_writer(logdir)
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)
        best_loss = float('inf')
        patience_counter = 0
        # profiler.start(logdir)
        for epoch in range(num_epochs):
            epoch_gen_loss, epoch_disc_loss = self.train_epoch(dataset)
            
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            print(f"Epoch {epoch}: Generator Loss = {avg_gen_loss}, Discriminator Loss = {avg_disc_loss}")
            # abs avg loss
            avg_disc_loss = abs(avg_disc_loss)
            # Early stopping logic remains the same...
            if avg_disc_loss < best_loss:
                best_loss = avg_disc_loss
                patience_counter = 0
                self.generator.save("generator.keras")
                self.critic.save("discriminator.keras")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter} / {patience}")
                if patience_counter >= patience:
                    print("Early stopping...")
                    break
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
