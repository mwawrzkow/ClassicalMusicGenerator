# Music Generation with GAN and RNN Networks
This repository contains a comparative study between **WSeqGAN-GP** (Wasserstein Sequence Generative Adversarial Network with Gradient Penalty) and **RNN** (Recurrent Neural Network) architectures for generating music sequences. The project is a part of my Master's thesis, focused on exploring the effectiveness of GANs and RNNs in music generation.
## Architectures

### WSeqGAN-GP with Gumbel Softmax
The WSeqGAN-GP architecture is designed to generate music sequences with a focus on handling pitch, step, duration, and velocity. The generator and discriminator (critic) are both based on LSTM layers, optimized using Adam optimizers. Gumbel softmax is used in the generator for categorical output on the pitch.

**Generator**
```python
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
```
**Critic**
```python
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
```

### RNN Architecture
The RNN architecture for music generation is based on LSTM layers. It generates the pitch, step, and duration of the notes, and optimizes the outputs using custom loss functions for each.

```python
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
            'duration': 1.0,
        },
        optimizer=optimizer)
    model.summary()
    return model
```

## How to Run the Project
### Linux 
Run the Docker container with the run.sh script on Linux. This script automatically detects whether the system has an NVIDIA or AMD GPU or neither and selects the appropriate TensorFlow image, and runs the container.
```bash
./run.sh
```

### Windows (CPU only)
On Windows, TensorFlow only supports CPU computation. Use the run.ps1 PowerShell script to build and run the Docker container.
```
./run.ps1
```

# Docker Build and Run
Both the run.sh (Linux) and run.ps1 (Windows) scripts handle the following tasks:

1. Build the Docker Image: The script builds the Docker image using the appropriate TensorFlow base image for the platform (GPU or CPU).

## Requirements 
* Docker 
* GPU support (optional for Linux, only CPU support on Windows)
* TensorFlow 2.x (handled within the Docker container)


# Conclusion
This project compares the effectiveness of GANs and RNNs in generating music sequences. By running the provided scripts, you can replicate the training process and generate music using both architectures. For more details on the implementation, refer to the `core_gan.py` and `core_rnn.py` files inside the `src/` directory.