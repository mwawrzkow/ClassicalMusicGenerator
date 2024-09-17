import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, GRU, LeakyReLU, Bidirectional, Conv1D, Lambda, LayerNormalization, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow_addons.layers import SpectralNormalization
import os

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
from base_model import BaseModel

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
if not tf.config.list_physical_devices('GPU'):
    print("No GPU found, Training might be really slow.")
    if os.environ.get("IGNORE_NO_GPU", False):
        pass
    raise RuntimeError("No GPU found, can't continue.")


tf.config.optimizer.set_jit("autoclustering")

logdir = None
tensorboard_callback = None
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1) distribution."""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=False):
    """Apply Gumbel-Softmax trick."""
    gumbel_noise = sample_gumbel(tf.shape(logits))
    y = logits + gumbel_noise
    y = tf.nn.softmax(y / temperature)

    if hard:
        
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
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
    
    pitch_min_idx = min_max['pitch_min']  
    pitch_max_idx = min_max['pitch_max']  
    mask = tf.concat([
        tf.fill([pitch_min_idx], 1e-9),  
        tf.fill([pitch_max_idx - pitch_min_idx + 1], 1.0),  
        tf.fill([128 - pitch_max_idx - 1], 1e-9)  
    ], axis=0)
    mask = tf.reshape(mask, (1, 1, 128))  
    pitch_probs = pitch_probs * mask
    pitch_probs = pitch_probs / tf.reduce_sum(pitch_probs, axis=-1, keepdims=True)
    return pitch_probs

class GAN(BaseModel):

    def __init__(self, input_dim, generator_output_dim, discriminator_output_dim, load=False, min_max= None, continue_training=False):
        self.rng = tf.random.Generator.from_seed(42)
        
        print(f"Available devices: {tf.config.list_physical_devices('GPU')}")
        self.generator = None
        self.critic = None
        
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
        
        self.load_for_training = load
        if self.load_for_training:
            if self.generator is None:
                print("Model not initialized, initializing model")
                self.generator = self.build_generator()
            if self.continue_training:
                self.generator.load_weights("generator.keras")
            if self.critic is None:
                print("Model not initialized, initializing model")
                self.critic = self.build_discriminator()
            if self.continue_training:
                self.critic.load_weights("discriminator.keras")
            print("Model loaded successfully.")            

        else: 
            self.generator = self.build_generator()
            self.critic = self.build_discriminator()
            self.generator.save_weights("generator.keras")


        
    def build_generator(self):
        inputs = Input(shape=(self.input_dim))
        x = LSTM(self.discriminator_output_dim[0] * 3 , return_sequences=True)(inputs)
        x = Dropout(0.4)(x)
        x = LSTM(self.discriminator_output_dim[0] * 3 , return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(self.discriminator_output_dim[0] * 3 , return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(self.discriminator_output_dim[0] * 3 , return_sequences=True)(x)
        x = Dropout(0.2)(x)
        pitch_output = Dense(128, activation=None, kernel_initializer='he_normal', name='pitch_output')(x)
        pitch_output = Lambda(lambda x: gumbel_softmax(x, temperature=1.0, hard=False))(pitch_output)
        pitch_output = Lambda(lambda x: apply_pitch_mask(x, { 'pitch_min': self.min_max['pitch_min'], 'pitch_max': self.min_max['pitch_max'] }))(pitch_output)
        duration_output = Dense(1, activation="sigmoid", kernel_initializer='he_normal', name='duration_output')(x)        
        step_output = Dense(1, activation="sigmoid", kernel_initializer='he_normal', name='step_output')(x)        
        velocity_output = Dense(1, activation="sigmoid", kernel_initializer='he_normal', name='velocity_output')(x)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.9)

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
        x = LSTM(self.discriminator_output_dim[0] * 4, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = LSTM(self.discriminator_output_dim[0] * 4 , return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='linear' ,kernel_initializer='he_normal')(x)  
        
        model = Model(inputs=[note_input, step_input, duration_input, velocity_input], outputs=x)
        model.summary()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.0, beta_2=0.9)
        model.compile(optimizer=self.critic_optimizer)
        if self.continue_training:
            model.load_weights("discriminator.keras")
        return model

    def generate_data(self, num_samples, temperature=1.0):
        """
        Generate fake MIDI data using the generator. Applies Gumbel-Softmax for pitch and scaling for step, duration, and velocity.
        """
        fake_data = self.generator(self.generate_noise(num_samples), training=False)
        fake_pitch_logits = fake_data['pitch'] 
        fake_pitch_probs = gumbel_softmax(fake_pitch_logits, temperature=temperature, hard=False)
        fake_pitch_notes = tf.random.categorical(tf.reshape(fake_pitch_probs, [-1, 128]), 1)
        fake_pitch_notes = tf.reshape(fake_pitch_notes, [num_samples, -1, 1]) 
        fake_pitch_notes = tf.cast(fake_pitch_notes, tf.float32)
        fake_step_scaled = self.min_max['step_min'] + fake_data['step'] * (self.min_max['step_max'] - self.min_max['step_min'])
        fake_duration_scaled = self.min_max['duration_min'] + fake_data['duration'] * (self.min_max['duration_max'] - self.min_max['duration_min'])
        fake_velocity_scaled = self.min_max['velocity_min'] + fake_data['velocity'] * (self.min_max['velocity_max'] - self.min_max['velocity_min'])
        fake_pitch_one_hot = tf.one_hot(tf.cast(fake_pitch_notes, tf.int32), depth=128, axis=-1) 
        fake_pitch_one_hot = tf.squeeze(fake_pitch_one_hot, axis=-2)
        generated_data = tf.concat([fake_pitch_notes, fake_step_scaled, fake_duration_scaled, fake_velocity_scaled], axis=-1)  
        return generated_data  
    
    def generate_noise(self, batch_size):
        combined_noise = self.rng.uniform([batch_size, self.input_dim[0], self.input_dim[1]])
        combined_noise += self.rng.normal([batch_size, self.input_dim[0], self.input_dim[1]], mean=0.0, stddev=0.01)
        combined_noise = tf.clip_by_value(combined_noise, 0.0, 1.0)
        
        return combined_noise

    @tf.function
    def gradient_penalty(self, real_data, fake_data):
        """
        Compute the gradient penalty for WGAN-GP.
        Inputs:
            real_data: Real samples (batch) in dict format {'pitch', 'step', 'duration', 'velocity'}
            fake_data: Fake samples (batch) in dict format {'pitch', 'step', 'duration', 'velocity'}
        """
        if real_data['pitch'].shape[0] != fake_data['pitch'].shape[0]:
            tf.print("Real data and fake data must have the same batch size but got {} and {}".format(real_data['pitch'].shape[0], fake_data['pitch'].shape[0]))
            raise ValueError("Real data and fake data must have the same batch size but got {} and {}".format(real_data['pitch'].shape[0], fake_data['pitch'].shape[0]))
        epsilon = tf.random.uniform([real_data['pitch'].shape[0]] + [1 for _ in range(len(real_data['pitch'].shape) - 1)], 0.0, 1.0)
        interpolated_data = {
            'pitch': epsilon * real_data['pitch'] + (1 - epsilon) * fake_data['pitch'],
            'step': epsilon * real_data['step'] + (1 - epsilon) * fake_data['step'],
            'duration': epsilon * real_data['duration'] + (1 - epsilon) * fake_data['duration'],
            'velocity': epsilon * real_data['velocity'] + (1 - epsilon) * fake_data['velocity']
        }
        with tf.GradientTape() as tape:
            tape.watch([interpolated_data['pitch'], interpolated_data['step'], interpolated_data['duration'], interpolated_data['velocity']])
            interpolated_scores = self.critic(interpolated_data, training=True)

        gradients = tape.gradient(interpolated_scores, [interpolated_data['pitch'], interpolated_data['step'], interpolated_data['duration'], interpolated_data['velocity']])

        grad_norm_pitch = tf.reduce_sum(tf.square(gradients[0]), axis=[1, 2]) 
        grad_norm_step = tf.reduce_sum(tf.square(gradients[1]), axis=[1, 2]) 
        grad_norm_duration = tf.reduce_sum(tf.square(gradients[2]), axis=[1, 2])
        grad_norm_velocity = tf.reduce_sum(tf.square(gradients[3]), axis=[1, 2])
        grad_norm = tf.sqrt(grad_norm_pitch + grad_norm_step + grad_norm_duration + grad_norm_velocity)

        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        return gradient_penalty

    @tf.function
    def compute_critic_loss(self, real_data, batch_size):
        """
        Compute the critic loss with gradient penalty.
        Inputs:
            real_data: Real samples from the dataset
            batch_size: Size of the batch
        """
        fake_data = self.generator(self.generate_noise(batch_size), training=False)
        scaled_step = self.min_max['step_min'] + fake_data['step'] * (self.min_max['step_max'] - self.min_max['step_min'])
        scaled_duration = self.min_max['duration_min'] + fake_data['duration'] * (self.min_max['duration_max'] - self.min_max['duration_min'])
        scaled_velocity = self.min_max['velocity_min'] + fake_data['velocity'] * (self.min_max['velocity_max'] - self.min_max['velocity_min'])
        
        fake_data_pitch_gumbel = gumbel_softmax(fake_data['pitch'], temperature=1.0, hard=True)  

        real_data_pitch = real_data[:, :, :128]  
        real_data_step = real_data[:, :, 128:129]  
        real_data_duration = real_data[:, :, 129:130]  
        real_data_velocity = real_data[:, :, 130:131]  

        scaled_generated_data = {
            'step': scaled_step,
            'duration': scaled_duration,
            'velocity': scaled_velocity,
            'pitch': fake_data_pitch_gumbel  
        }
        real_data ={ 'step': real_data_step, 'duration': real_data_duration, 'velocity': real_data_velocity, 'pitch': real_data_pitch}
        real_scores = self.critic(real_data, training=True)
        fake_scores = self.critic(scaled_generated_data, training=True)
        d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        gp = self.gradient_penalty(real_data, scaled_generated_data)
        total_d_loss = d_loss + 10.0 * gp  

        return total_d_loss
    
    @tf.function
    def train_critic_step(self, real_data, batch_size, grad_norm_th=7.0):
        """
        Training step for the discriminator with gradient penalty and gradient clipping.
        Inputs:
            real_data: Real samples from the dataset
            batch_size: Batch size for training
            grad_norm_th: Threshold for gradient clipping
        """
        with tf.GradientTape() as disc_tape:
            disc_loss = self.compute_critic_loss(real_data, batch_size)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.critic.trainable_variables)
        gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, grad_norm_th)
        self.critic_optimizer.apply_gradients(zip(gradients_of_discriminator, self.critic.trainable_variables))
        return disc_loss
    
    @tf.function
    def train_generator_real(self, batch_size): 
            with tf.GradientTape() as tape:
                generated_data = self.generator(self.generate_noise(batch_size), training=True)
                scaled_step = self.min_max['step_min'] + generated_data['step'] * (self.min_max['step_max'] - self.min_max['step_min'])
                scaled_duration = self.min_max['duration_min'] + generated_data['duration'] * (self.min_max['duration_max'] - self.min_max['duration_min'])
                scaled_velocity = self.min_max['velocity_min'] + generated_data['velocity'] * (self.min_max['velocity_max'] - self.min_max['velocity_min'])
                generated_pitch = gumbel_softmax(generated_data['pitch'], temperature=1.0, hard=True)
                scaled_generated_data = {
                    'step': scaled_step,
                    'duration': scaled_duration,
                    'velocity': scaled_velocity,
                    'pitch': generated_pitch  
                }
                generated_scores = self.critic(scaled_generated_data, training=False)
                generator_loss = -tf.reduce_mean(generated_scores)
                
                total_generator_loss =  generator_loss 
            gen_gradients = tape.gradient(total_generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables)) 
            return generator_loss
        

    def train_step(self, real_data, batch_size=4, is_generator_step=False, pretrain_critic=False):
        crit_losses = self.train_critic_step(real_data, batch_size)
        gen_loss = None
        if is_generator_step and not pretrain_critic:
            gen_loss = self.train_generator_real(batch_size)
        return gen_loss, crit_losses
    
    def train_epoch(self, dataset: tf.data.Dataset, batch_size=4, critic_giga_steps=5, pretrain_critic=False):
        epoch_gen_loss = []
        epoch_disc_loss = []
        step = 0
        for real_batch in dataset:
            is_generator_step = step % critic_giga_steps == 0
            gen_loss, disc_loss = self.train_step(real_batch, batch_size, is_generator_step, pretrain_critic)
            if gen_loss is not None:
                epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss) 
            step += 1
            if pretrain_critic == False: 
                print(f"Step {step}: Generator Loss = {epoch_gen_loss[-1] }, Discriminator Loss = {epoch_disc_loss[-1]}")
            else:
                print(f"Step {step}: Dyscriminator headstart Discriminator Loss = {epoch_disc_loss[-1]}")
                    
        return epoch_gen_loss, epoch_disc_loss, step
    
    def train(self, dataset, num_epochs, patience=3, batch_size=4, continue_training=False, callback = None):
        if not os.path.exists("gan_checkpoints"):
            os.mkdir("gan_checkpoints")
        if self.load_for_training and not continue_training:
            print("Model loaded, skipping training")
            return
        logdir = "src/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(logdir)
        self.train_step_counter = tf.Variable(0, dtype=tf.int64)
        patience_counter = 0
        generator_generated_data = []
        
        critic_giga_steps = 3
        idx = 0 
        print(f"Training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            pretrain_critic = idx < 5
            idx += 1
            start_time = datetime.now()
            epoch_gen_loss, epoch_disc_loss, steps = self.train_epoch(dataset, batch_size,critic_giga_steps, pretrain_critic=pretrain_critic)
            callback(self, epoch)
            end_time = datetime.now()
            self.train_summary_writer.flush()
            print(f"Epoch {epoch} took {end_time - start_time}, possible train remaining time: {(end_time - start_time) * (num_epochs - epoch)}")
            
            avg_gen_loss = -1
            if len(epoch_gen_loss) > 0: 
                avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            with self.train_summary_writer.as_default(): 
                tf.summary.scalar('Critic Loss epoch', avg_disc_loss, step=self.train_step_counter)
                tf.summary.scalar('Generator Loss epoch', avg_gen_loss, step=self.train_step_counter)                   
            self.train_step_counter.assign_add(1)
            print(f"Epoch {epoch}: Generator Loss = {avg_gen_loss}, Discriminator Loss = {avg_disc_loss}, Steps = {steps}")
            critic_giga_steps = 5 if avg_disc_loss < -2.0 else 1
            if avg_disc_loss < -2.0:
                print("Error in training, has become too sure of itself. Model propably collapsed, thus incrementing patience")
                patience_counter += 1
            else: 
                patience_counter = 0
            
            if avg_disc_loss < -4 and not pretrain_critic: 
                print("Propably Generator dominated, critic. Early stopping now")
                patience_counter = patience
                
            avg_disc_loss = abs(avg_disc_loss)
            
            if patience_counter >= patience and not pretrain_critic:
                print("Early stopping")
                break
            if idx % 10 == 0:
                self.generator.save_weights(f"gan_checkpoints/generator_{idx}.keras")
                self.critic.save_weights(f"gan_checkpoints/discriminator_{idx}.keras")
 
        self.generator.save_weights("generator.keras")
        self.critic.save_weights("discriminator.keras")
        return generator_generated_data
    
    def save(self, path):
        self.generator.save_weights(path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        generator = path + "/generator.keras"
        
        if self.generator is None:
            print(f"Model not initialized, initializing model")
            self.critic = self.build_generator()
        self.generator.load_weights(generator)
        if self.critic is None:
            print(f"Model not initialized, initializing model")
            self.critic = self.build_discriminator()
        critic = path + "/discriminator.keras"
        
        if not os.path.exists(critic):
            print(f"No weights found for the critic, skipping loading")
        else: 
            self.critic.load_weights(critic)
        print(f"Model loaded from {path}")
