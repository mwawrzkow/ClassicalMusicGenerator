if __name__ == '__main__':
    print("this file is not meant to be run directly")
    print('Please run the __main__.py file or the directory containing the __main__.py file')
    sys.exit(1)

import argparse
import collections
from copy import deepcopy
import glob
import random
import pretty_midi
import pandas as pd
import numpy as np
import pathlib 
import concurrent.futures
import os 
import tensorflow as tf
from tqdm import tqdm
import sys 
from network_gan import GAN
from utils import cls
from midi import Midis
import matplotlib.pyplot as plt
key_order = ['pitch', 'step', 'duration', 'velocity']

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  if len(pm.instruments) == 0:
    raise ValueError(f"No instruments found in {midi_file}")
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    notes['velocity'].append(note.velocity)
    prev_start = start
  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
def load_all_midi_files(directory: str, num_files: int) -> tf.data.Dataset:
    min_max = {
    'step_min': float('inf'),
    'step_max': float('-inf'),
    'pitch_min': float('inf'),
    'pitch_max': float('-inf'),
    'duration_min': float('inf'),
    'duration_max': float('-inf'),
    'velocity_min': float('inf'),
    'velocity_max': float('-inf')
    }

    def update_min_max(notes):
        # Update min/max values based on the notes DataFrame
        min_max['pitch_min'] = min(min_max['pitch_min'], notes['pitch'].min())
        min_max['pitch_max'] = max(min_max['pitch_max'], notes['pitch'].max())
        min_max['step_min'] = min(min_max['step_min'], notes['step'].min())
        min_max['step_max'] = max(min_max['step_max'], notes['step'].max())
        min_max['duration_min'] = min(min_max['duration_min'], notes['duration'].min())
        min_max['duration_max'] = max(min_max['duration_max'], notes['duration'].max())
        min_max['velocity_min'] = min(min_max['velocity_min'], notes['velocity'].min())
        min_max['velocity_max'] = max(min_max['velocity_max'], notes['velocity'].max())

    filenames = glob.glob(str(directory/'*.mid*'))
    all_notes = []
    num_files = len(filenames) if num_files is None else num_files
    i = 0 
    if num_files > 30:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for f in tqdm(filenames[:num_files], desc="Processing MIDI files"):
                try:
                    future = executor.submit(midi_to_notes, f)
                    futures.append(future)
                except Exception as e:
                    print(f"Error processing {f}: {e}")

            all_notes = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    notes = future.result()
                    all_notes.append(notes)
                    update_min_max(notes)
                    print(f"Processed {i}/{num_files} files")
                    i += 1
                except Exception as e:
                    print(f"Error processing MIDI file: {e}")
    else:
        for f in filenames[:num_files]:
            try:
                notes = midi_to_notes(f)
                all_notes.append(notes)
                update_min_max(notes)
                print(f"Processed {i}/{num_files} files")
                i += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)
    data = np.stack([all_notes[key] for key in key_order], axis=1)
    data = np.array(data)
    data = data.astype(np.float32)
    # normalize note pitch
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset, len(data), min_max
import tensorflow as tf

def create_sequences_tf(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size=128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequences with no labels."""

    # Adjust seq_length to include the full sequence length
    seq_length = seq_length

    # Create sliding windows of sequence length across the dataset
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def one_hot_and_normalize(x):
        pitch = x[:, 0]  # Extract pitch values (shape: [seq_length])
        step = x[:, 1]   # Extract step values (shape: [seq_length])
        duration = x[:, 2]  # Extract duration values (shape: [seq_length])
        velocity = x[:, 3]  # Extract velocity values (shape: [seq_length])
        # One-hot encode the pitch values
        pitch_one_hot = tf.one_hot(tf.cast(pitch, tf.int32), depth=vocab_size)

        # Stack pitch (one-hot), step, and duration into the same tensor
        step = tf.expand_dims(step, -1)  # Ensure step has shape [seq_length, 1]
        duration = tf.expand_dims(duration, -1)  # Ensure duration has shape [seq_length, 1]
        velocity = tf.expand_dims(velocity, -1)  # Ensure velocity has shape [seq_length, 1]

        # Concatenate along the last axis: [seq_length, pitch_depth + 1 + 1]
        return tf.concat([pitch_one_hot, step, duration, velocity], axis=-1)

    # Return scaled sequences
    return sequences.map(one_hot_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)



def generate_sequence(model: GAN, seed_sequence: int, length:int, filename: str,  temperature=2.0):

    temperature = 1.0

    data = model.generate_data(1, temperature).numpy()

    # assume data in format         fake_data = tf.concat([fake_data['pitch'], fake_data['step'], fake_data['duration']], axis=-1)
    # denormalize pitch
    # data = data * np.array([128, 1, 1])
    notes = [] 
    for i in range(len(data[0])):
        notes.append({key_order[j]: data[0][i][j] for j in range(len(key_order))})
    data = pd.DataFrame(notes, columns=['pitch', 'step', 'duration', 'velocity'])
    
    return data 

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  generated_notes = []
#   get first row
  
  for i, note in notes.iterrows():
    start = float(prev_start + (note['step']))
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=int(note['velocity']),
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    generated_notes.append(deepcopy(note))
    instrument.notes.append(note)
    prev_start = start

#   save create plot with the notes
# Create a plot of the notes
  start_times = [note.start for note in generated_notes]
  pitches = [note.pitch for note in generated_notes]
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(start_times, pitches, c='blue', label='Notes')
  ax.set_xlabel('Start Time')
  ax.set_ylabel('Pitch')
  ax.set_title('Generated Notes')
  ax.legend()
  plt.savefig(out_file.replace('.mid', '.png'))
  plt.close()
  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def inspect_batch(dataset, seq_length, output_dir):
    # Take one batch from the dataset
    for batch in dataset.take(1):
        print("Batch shape:", batch.shape)
        print("First sequence in the batch:")
        print(batch[0].numpy())  # Convert to numpy for easy inspection

        # Check normalization range
        print("Step range in batch:", batch[:, :, 128].numpy().min(), batch[:, :, 1].numpy().max())
        print("Duration range in batch:", batch[:, :, 129].numpy().min(), batch[:, :, 2].numpy().max())

        # Validate the sequence length
        assert batch.shape[1] == seq_length, f"Expected sequence length {seq_length}, but got {batch.shape[1]}"

        # Validate that pitch is in range 0 to 1 (after normalization)
        # assert batch[:, :, 0].numpy().min() >= 0 and batch[:, :, 0].numpy().max() <= 1, "Pitch values out of expected range [0, 1]"

        # get notes from hot encoded data
        sequence = batch[0]  # Shape: [seq_length, 130]

        # Extract pitch, step, and duration from the selected sequence
        pitch = sequence[:, :128]  # One-hot encoded pitch
        pitch = tf.argmax(pitch, axis=-1).numpy()  # Get the actual pitch value (convert to numpy)
        step = sequence[:, 128].numpy()  # Step value
        duration = sequence[:, 129].numpy()  # Duration value
        velocity = sequence[:, 130].numpy()  # Velocity value

        # Stack pitch, step, and duration together
        melody = np.stack([pitch, step, duration, velocity], axis=-1)  # Shape: [seq_length, 3]
        
        # Convert to DataFrame
        notes_df = pd.DataFrame(melody, columns=['pitch', 'step', 'duration', 'velocity'])

        # Optionally, denormalize the pitch to MIDI range (0 to 127)
        # notes_df['pitch'] = notes_df['pitch'] * 128
        # notes_df['pitch'] = notes_df['pitch'].astype(int)  # Ensure pitch values are integers

        # Save the sequence to a MIDI file
        output_file = os.path.join(output_dir, 'validation_output.mid')
        notes_to_midi(notes_df, output_file, 'Acoustic Grand Piano')

        print(f"Saved validation output to {output_file}")

def getMinMaxValues(dataset, vocab_size=128):
    """
    Calculate the min and max values for one-hot encoded pitch, step, duration, and velocity from the dataset.

    Arguments:
    - dataset: A TensorFlow dataset containing the music data.
    - vocab_size: The vocabulary size for one-hot encoded pitch (default: 128).

    Returns:
    - A dictionary with min and max values for pitch, step, duration, and velocity.
    """
    min_max = {
        'step_min': float('inf'),
        'step_max': float('-inf'),
        'pitch_min': float('inf'),
        'pitch_max': float('-inf'),
        'duration_min': float('inf'),
        'duration_max': float('-inf'),
        'velocity_min': float('inf'),
        'velocity_max': float('-inf')
    }

    for batch in tqdm(dataset, desc="Calculating fine tuning parameters"):
        # One-hot encoded pitch is [batch_size, seq_length, vocab_size]
        pitch_one_hot = batch[:, :, :vocab_size]
        pitch = tf.argmax(pitch_one_hot, axis=-1).numpy()  # Convert one-hot to pitch indices

        # Extract other features (step, duration, velocity)
        step = batch[:, :, vocab_size].numpy()
        duration = batch[:, :, vocab_size + 1].numpy()
        velocity = batch[:, :, vocab_size + 2].numpy()

        # Calculate min/max for pitch
        pitch_min = np.min(pitch)
        pitch_max = np.max(pitch)

        # Calculate min/max for step, duration, and velocity
        step_min = np.min(step)
        step_max = np.max(step)
        duration_min = np.min(duration)
        duration_max = np.max(duration)
        velocity_min = np.min(velocity)
        velocity_max = np.max(velocity)

        # Update the min/max values in the dictionary
        min_max['step_min'] = min(min_max['step_min'], step_min)
        min_max['step_max'] = max(min_max['step_max'], step_max)
        min_max['pitch_min'] = min(min_max['pitch_min'], pitch_min)
        min_max['pitch_max'] = max(min_max['pitch_max'], pitch_max)
        min_max['duration_min'] = min(min_max['duration_min'], duration_min)
        min_max['duration_max'] = max(min_max['duration_max'], duration_max)
        min_max['velocity_min'] = min(min_max['velocity_min'], velocity_min)
        min_max['velocity_max'] = max(min_max['velocity_max'], velocity_max)

    return min_max

def generator_callback(args):
    if not os.path.exists("generator_training"):
        os.mkdir("generator_training")
    def call(gan, epoch): 
        seed_sequence = None
        generated_notes = generate_sequence(gan, seed_sequence, args.length, None)
        notes_to_midi(generated_notes, f"generator_training/generated_epoch_{epoch}.mid", "Acoustic Grand Piano")
    return call 
def run(args):
    # cls()
    print("selected gan")
    path = pathlib.Path(args.dataset)
    g_o_d = (args.seq_length,3)
    d_i_d = deepcopy(g_o_d)
    print("Loading MIDI files...")
    dataset, size, min_max = load_all_midi_files(pathlib.Path(args.dataset), args.num_files)
    tf_dataset = create_sequences_tf(dataset, seq_length=args.seq_length)
    print("Shuffling dataset...")
    dataset = tf_dataset.shuffle(buffer_size=size // args.seq_length).batch(args.batch_size, drop_remainder=True)
    print("Loaded dataset")
    print("Min and Max values:")
    print("Step Min:", min_max['step_min'], "Step Max:", min_max['step_max'])
    print("Pitch Min:", min_max['pitch_min'], "Pitch Max:", min_max['pitch_max'])
    print("Duration Min:", min_max['duration_min'], "Duration Max:", min_max['duration_max'])
    print("Velocity Min:", min_max['velocity_min'], "Velocity Max:", min_max['velocity_max'])
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        gan = GAN((args.seq_length,3),g_o_d, d_i_d, True, min_max, args.continue_training)
        gan.load(args.checkpoint)
        # gan.load(args.checkpoint)
        # gan.load("./")
        print("Loaded checkpoint")
    else: 
        # take one from dataset and save it to midi to check if it is correct
        inspect_batch(dataset, args.seq_length, args.output)
        gan = GAN((args.seq_length,3),g_o_d, d_i_d, args.continue_training, min_max, args.continue_training)
        gan.train(dataset, args.epochs, batch_size=args.batch_size, continue_training=args.continue_training, callback=generator_callback(args))
        gan.save('model_data/gan.keras')
    midi_files = glob.glob(str(path/'*.mid*'))
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # gan = GAN((args.seq_length,3),g_o_d, d_i_d, True)
    for n in range(args.num_generations):
        print(f"Generating sequence {n+1}...")
        random_file = random.choice(midi_files)
        seed_sequence = midi_to_notes(random_file)
        generated_notes = generate_sequence(gan, seed_sequence, args.length, random_file)
        print(f"Generated sequence {n+1}")
        notes_to_midi(generated_notes, f"{args.output}/generated_{n+1}.mid", "Acoustic Grand Piano")