import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class GAN:
    """
    Generative Adversarial Network (GAN) class.
    """

    def __init__(self, input_dim, generator_output_dim, discriminator_output_dim):
        """
        Constructor for the GAN class.

        Parameters:
        - input_dim (int): Dimension of the input noise vector.
        - generator_output_dim (int): Dimension of the generator output.
        - discriminator_output_dim (int): Dimension of the discriminator output.
        """
        # print available devices
        print(f"Available devices: {tf.config.list_physical_devices('GPU')}")

        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim
        self.discriminator_output_dim = discriminator_output_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()
        
    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_dim=self.input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        # model.add(layers.Dense(64, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dense(4, activation='sigmoid'))
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_shape=(4,), activation='relu'))
        # model.add(layers.Dropout(0.3))
        # model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    def generate_data(self, num_samples):
        # generate only data which is not detected as fake by the discriminator
        noise = tf.random.normal([num_samples, self.input_dim])
        generated_data = self.generator(noise)
            # check if discriminator mostly thinks that the data is real
        return generated_data
        
    def generate_real_batch(self, real_data, batch_size):
        return [real_data[i:i + batch_size] for i in range(0, len(real_data), batch_size)]

    @tf.function
    def train_step(self, current_batch_size, batch_size, real_batch):
        noise = tf.random.normal([current_batch_size, self.input_dim])

        # Train discriminator
        generated_data = self.generator(noise)
        real_labels = tf.ones((current_batch_size, 1))
        generated_labels = tf.zeros((current_batch_size, 1))
            
        with tf.GradientTape() as disc_tape:
            real_predictions = self.discriminator(real_batch)
            generated_predictions = self.discriminator(generated_data)
            disc_loss = self.discriminator_loss(real_predictions, generated_predictions, real_labels, generated_labels)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        # Train generator
        noise = tf.random.normal([current_batch_size, self.input_dim])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise)
            generated_predictions = self.discriminator(generated_data)
            gen_loss = self.generator_loss(generated_predictions, real_labels)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss, disc_loss
                

    def train(self, real_data, num_epochs, batch_size):
        real_batches = self.generate_real_batch(real_data, batch_size)
        csv = []
        print("Number of batches: {}".format(len(real_batches)))
        print("Number of epochs: {}".format(num_epochs))
        print("Starting training...")
        for epoch in range(num_epochs):
            gen_loss = 0
            disc_loss = 0
            for real_batch in real_batches:
                current_batch_size = len(real_batch)
                gen_loss, disc_loss = self.train_step(current_batch_size, batch_size, real_batch)
            if epoch % 10 == 0:
                print("Epoch: {}, {} percent epoch done, Generator loss: {}, Discriminator loss: {}".format(epoch, round(epoch/num_epochs, 2) , gen_loss, disc_loss))
            csv.append([epoch, gen_loss, disc_loss])
        return np.array(csv)
    
    def discriminator_loss(self, real_predictions, generated_predictions, real_labels, generated_labels):
        real_loss = tf.keras.losses.BinaryCrossentropy()(real_labels, real_predictions)
        generated_loss = tf.keras.losses.BinaryCrossentropy()(generated_labels, generated_predictions)
        total_loss = real_loss + generated_loss
        return total_loss
    
    def generator_loss(self, generated_predictions, real_labels):
        return tf.keras.losses.BinaryCrossentropy()(real_labels, generated_predictions)
