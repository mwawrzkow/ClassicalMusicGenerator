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
key_order = ['pitch', 'step', 'duration']

def normalize_data(data, min_value=0, max_value=127, scale_min=-1, scale_max=1):
    """Normalize data to be between scale_min and scale_max."""
    data_min = np.min(data)
    data_max = np.max(data)
    return scale_min + (data - data_min) * (scale_max - scale_min) / (data_max - data_min)

def denormalize_data(data, original_min=0, original_max=127, scale_min=-1, scale_max=1):
    """Denormalize data from scale_min and scale_max back to original range."""
    return original_min + (data - scale_min) * (original_max - original_min) / (scale_max - scale_min)

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
    prev_start = start
  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
def load_all_midi_files(directory: str, num_files: int) -> tf.data.Dataset:
    filenames = glob.glob(str(directory/'*.mid*'))
    all_notes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for f in tqdm(filenames[:num_files], desc="Processing MIDI files"):
            try:
                future = executor.submit(midi_to_notes, f)
                futures.append(future)
            except Exception as e:
                print(f"Error processing {f}: {e}")

        all_notes = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing MIDI files"):
            try:
                notes = future.result()
                all_notes.append(notes)
            except Exception as e:
                print(f"Error processing MIDI file: {e}")
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)
    data = np.stack([all_notes[key] for key in key_order], axis=1)
    data = np.array(data)
    # normalize note pitch
    data[:, 0] = normalize_data(data[:, 0])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset, len(data)
def create_sequences_tf(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Define key order for labels

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def generate_sequence(model: GAN, seed_sequence: int, length:int, filename: str,  temperature=2.0):
    raw_notes = midi_to_notes(filename)

    temperature = 3.0
    num_predictions = 1000

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = (
        sample_notes[:length] / np.array([128, 1, 1]))

    generated_notes = []
    prev_start = 0
    data = model.generate_data(1).numpy()

    # denormalize pitch
    # data[:, 0] = denormalize_data(data[:, 0])
    # abs on pitch
    # data[:, 0] = np.abs(data[:, 0])

    # Ensure the data is properly reshaped/processed if necessary
    # For example, if data comes out as (512, sequence_length, 3)
    # Ensure it is correctly handled

    notes = [] 
    for i in range(len(data[0])):
        notes.append({key: data[0][i][j] for j, key in enumerate(key_order)})

    data = pd.DataFrame(notes, columns=['pitch', 'step', 'duration'])
    
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
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    generated_notes.append(deepcopy(note))
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


def run(args):
    cls()
    print("selected gan")
    path = pathlib.Path(args.dataset)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        gan = GAN((args.seq_length,3), 4)        
        gan.load(args.checkpoint)
        print("Loaded checkpoint")
    else: 
        print("Loading MIDI files...")
        dataset, size = load_all_midi_files(pathlib.Path(args.dataset), args.num_files)
        tf_dataset = create_sequences_tf(dataset, seq_length=args.seq_length)
        print("Loaded dataset")
        print("Shuffling dataset...")
        dataset = tf_dataset.shuffle(buffer_size=size // args.seq_length).batch(args.batch_size, drop_remainder=True)
        g_o_d = (args.seq_length,3)
        d_i_d = deepcopy(g_o_d)
        gan = GAN((args.seq_length,3),g_o_d, d_i_d, False)
        gan.train(dataset, args.epochs)
        gan.save('model_data/rnn.keras')
    midi_files = glob.glob(str(path/'*.mid*'))
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for n in range(args.num_generations):
        print(f"Generating sequence {n+1}...")
        random_file = random.choice(midi_files)
        seed_sequence = midi_to_notes(random_file)
        generated_notes = generate_sequence(gan, seed_sequence, args.length, random_file)
        print(f"Generated sequence {n+1}")
        notes_to_midi(generated_notes, f"output/generated_{n+1}.mid", "Acoustic Grand Piano")