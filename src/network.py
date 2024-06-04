import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Bidirectional, BatchNormalization, Reshape, Input, Flatten
from tqdm import tqdm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("No GPU found, can't continue.")
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
# tf.config.optimizer.set_jit(True)
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
            self.discriminator = tf.keras.models.load_model("./discriminator.keras")
            print("Model loaded successfully.")
            self.generator.summary()
            self.discriminator.summary()
            # should we compile the model again?

        else: 
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
            
            # self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            self.generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, beta_1=0.5, beta_2=0.98)
            self.generator.compile(optimizer=self.generator_optimizer)
            self.discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
            self.discriminator.compile(optimizer=self.discriminator_optimizer)
        
    def build_generator(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim)))  # Adding sequence length of 1
        model.add(Dense(512))
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(Bidirectional(LSTM(128, return_sequences=False)))
        model.add(Dense(256, activation='relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
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
        model.add(Dense(512))
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # model.add(Bidirectional(LSTM(128, return_sequences=False)))
        model.add(Dense(256, activation='relu'))
        model.add(LeakyReLU(alpha=0.2))
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
        model.add(Dense(1, activation='sigmoid'))
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
        # check if discriminator mostly thinks that the data is real
        predictions = self.discriminator.predict(generated_data)
        # if the discriminator is not sure, generate new data
        tries = 0
        MAX_TRIES = 100
        while tf.reduce_mean(predictions) < 0.9:
            noise = self.generate_noise(num_samples)
            generated_data = self.generator.predict(noise)
            predictions = self.discriminator.predict(generated_data)
            # print predictions
            # print(f"Discriminator predictions: {predictions}")
            tries += 1
            if tries > MAX_TRIES:
                print(f"Discriminator is not sure, giving up after {MAX_TRIES} tries.")
                break
            else: 
                print(f"Discriminator is not sure, trying again. Try number: {tries}")
        return generated_data
        
    def generate_real_batch(self, real_data, batch_size):
        batched_data = [real_data[i:i + batch_size] for i in range(0, len(real_data), batch_size)]
    # # Ensure each batch is correctly shaped
        batched_data = [batch.reshape(-1, batch_size, 1) for batch in batched_data if batch.shape[0] == batch_size]
        return  batched_data
    def generate_noise(self, batch_size):
        # Generating noise with the correct shape for the generator
        return tf.random.normal([batch_size, *self.input_dim])
            
        return epoch_gen_loss, epoch_disc_loss
    @tf.function
    def train_step(self, real_batch, train_only_discriminator=False):
        current_batch_size = tf.shape(real_batch)[0]
        
        noise = self.generate_noise(current_batch_size)
        generated_data = None 
        gen_loss = None
        if not train_only_discriminator:
            with tf.GradientTape() as gen_tape:
                generated_data = self.generator(noise, training=True)
                # Reuse generated data and compute new predictions inside the generator tape
                generated_predictions = self.discriminator(generated_data)
                gen_labels = tf.ones_like(generated_predictions) 
                gen_loss = self.generator_loss(generated_predictions, gen_labels)
        else: 
            generated_data = self.generator(noise, training=False)

        # print("Shape of generator output:", generated_data.shape)
        # Apply label smoothing here
        # generate random number between 0.9 and 1.3 
        positive_random = tf.random.uniform((1,), minval=0.7, maxval=1)
        negative_random = tf.random.uniform((1,), minval=0.0, maxval=0.3)
        smooth_positive_labels = tf.ones((current_batch_size, 1)) * positive_random  # Smoothing for positive labels
        smooth_negative_labels = tf.zeros((current_batch_size, 1)) * negative_random  # Smoothing for negative labels
        # real_batch = (current_batch_size, self.discriminator_output_dim)
        
        with tf.GradientTape() as disc_tape:
            real_predictions = self.discriminator(real_batch, training=True)
            generated_predictions = self.discriminator(generated_data, training=True)
            disc_loss = self.discriminator_loss(real_predictions, generated_predictions, smooth_positive_labels, smooth_negative_labels)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        if not train_only_discriminator:
            gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss, disc_loss


    def train_epoch(self, dataset: tf.data.Dataset):
        epoch_gen_loss = []
        epoch_disc_loss = []
        SKIP_GENERATOR = 10
        for idx, real_batch in enumerate(tqdm(dataset.prefetch(tf.data.experimental.AUTOTUNE))):
            # every 10 steps train both generator and discriminator
            # if idx % SKIP_GENERATOR == 0:
            gen_loss, disc_loss = self.train_step(real_batch)
            # else:
            #     gen_loss, disc_loss = self.train_step(real_batch, train_only_discriminator=True)
            # Log every 1000 steps
            if idx % SKIP_GENERATOR == 0:
                epoch_gen_loss.append(gen_loss)
                epoch_disc_loss.append(disc_loss)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("Generator Loss", gen_loss, step=self.train_step_counter)
                    tf.summary.scalar("Discriminator Loss", disc_loss, step=self.train_step_counter)
            
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
        for epoch in range(num_epochs):
            epoch_gen_loss, epoch_disc_loss = self.train_epoch(dataset)
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            print(f"Epoch {epoch}: Generator Loss = {avg_gen_loss}, Discriminator Loss = {avg_disc_loss}")

            # Early stopping logic remains the same...
            if avg_disc_loss < best_loss:
                best_loss = avg_disc_loss
                patience_counter = 0
                self.generator.save("generator.keras")
                self.discriminator.save("discriminator.keras")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter} / {patience}")
                if patience_counter >= patience:
                    print("Early stopping...")
                    break
    
    def discriminator_loss(self, real_predictions, generated_predictions, real_labels, generated_labels):
        real_loss = tf.keras.losses.BinaryCrossentropy()(real_labels, real_predictions)
        generated_loss = tf.keras.losses.BinaryCrossentropy()(generated_labels, generated_predictions)
        total_loss = real_loss + generated_loss
        return total_loss
    
    def generator_loss(self, generated_predictions, real_labels):
        return tf.keras.losses.BinaryCrossentropy()(real_labels, generated_predictions)
