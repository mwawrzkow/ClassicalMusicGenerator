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
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset, len(data)
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

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Return scaled sequences
    return sequences.map(scale_pitch, num_parallel_calls=tf.data.AUTOTUNE)



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

    # assume data in format         fake_data = tf.concat([fake_data['pitch'], fake_data['step'], fake_data['duration']], axis=-1)
    # denormalize pitch
    # data = data * np.array([128, 1, 1])
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
#   get first row
  
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

def inspect_batch(dataset, seq_length, output_dir):
    # Take one batch from the dataset
    for batch in dataset.take(1):
        print("Batch shape:", batch.shape)
        print("First sequence in the batch:")
        print(batch[0].numpy())  # Convert to numpy for easy inspection

        # Check normalization range
        print("Pitch range in batch:", batch[:, :, 0].numpy().min(), batch[:, :, 0].numpy().max())
        print("Step range in batch:", batch[:, :, 1].numpy().min(), batch[:, :, 1].numpy().max())
        print("Duration range in batch:", batch[:, :, 2].numpy().min(), batch[:, :, 2].numpy().max())

        # Validate the sequence length
        assert batch.shape[1] == seq_length, f"Expected sequence length {seq_length}, but got {batch.shape[1]}"

        # Validate that pitch is in range 0 to 1 (after normalization)
        # assert batch[:, :, 0].numpy().min() >= 0 and batch[:, :, 0].numpy().max() <= 1, "Pitch values out of expected range [0, 1]"

        # Convert the first sequence in the batch to MIDI format
        notes_df = pd.DataFrame(batch[0].numpy(), columns=['pitch', 'step', 'duration'])

        # Denormalize the pitch to MIDI range (0 to 127)
        notes_df['pitch'] = notes_df['pitch'] * 128
        notes_df['pitch'] = notes_df['pitch'].astype(int)  # Ensure pitch values are integers

        # Save the sequence to a MIDI file
        output_file = os.path.join(output_dir, 'validation_output.mid')
        notes_to_midi(notes_df, output_file, 'Acoustic Grand Piano')

        print(f"Saved validation output to {output_file}")

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
        # take one from dataset and save it to midi to check if it is correct
        inspect_batch(dataset, args.seq_length, args.output)
        g_o_d = (args.seq_length,3)
        d_i_d = deepcopy(g_o_d)
        gan = GAN((args.seq_length,3),g_o_d, d_i_d, False)
        gan.train(dataset, args.epochs, batch_size=args.batch_size)
        gan.save('model_data/gan.keras')
    midi_files = glob.glob(str(path/'*.mid*'))
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for n in tqdm(range(args.num_generations), desc="Generating MIDI files"):
        print(f"Generating sequence {n+1}...")
        random_file = random.choice(midi_files)
        seed_sequence = midi_to_notes(random_file)
        generated_notes = generate_sequence(gan, seed_sequence, args.length, random_file)
        print(f"Generated sequence {n+1}")
        notes_to_midi(generated_notes, f"{args.output}/generated_{n+1}.mid", "Acoustic Grand Piano")