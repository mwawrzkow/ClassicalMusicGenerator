if __name__ == '__main__':
    print("this file is not meant to be run directly")
    print('Please run the __main__.py file or the directory containing the __main__.py file')
    sys.exit(1)

import argparse
import collections
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
from network_rrn import RNNModel
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
        for f in filenames[:num_files]:
            try:
                future = executor.submit(midi_to_notes, f)
                futures.append(future)
            except Exception as e:
                print(f"Error processing {f}: {e}")

        all_notes = []
        i = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                notes = future.result()
                all_notes.append(notes)
                print(f"Processed {i}/{num_files} files")
                i += 1
            except Exception as e:
                print(f"Error processing MIDI file: {e}")
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)
    data = np.stack([all_notes[key] for key in key_order], axis=1)
    data = np.array(data)
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
def generate_sequence(model: RNNModel, seed_sequence: int, length:int, filename: str,  temperature=2.0):
    raw_notes = midi_to_notes(filename)

    temperature = 3.0
    num_predictions = 100

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = (
        sample_notes[:length] / np.array([128, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in tqdm(range(num_predictions), desc="Generating Notes"):
        pitch, step, duration = model.predict_next_note(input_notes,temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start
    generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))
    return generated_notes
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
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

def run(args): 
    # cls()
    print("selected rnn")
    path = pathlib.Path(args.dataset)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        rnn = RNNModel((args.seq_length,3), 4)        
        rnn.load(args.checkpoint)
        print("Loaded checkpoint")
    else: 
        print("Loading MIDI files...")
        dataset, size = load_all_midi_files(pathlib.Path(args.dataset), args.num_files)
        tf_dataset = create_sequences_tf(dataset, seq_length=args.seq_length)
        print("Loaded dataset")
        print("Shuffling dataset...")
        dataset = tf_dataset.shuffle(buffer_size=size // args.seq_length).batch(args.batch_size, drop_remainder=True)
        rnn = RNNModel((args.seq_length,3), 4)
        rnn.train(dataset, args.epochs)
        rnn.save('model_data/rnn.keras')
    midi_files = glob.glob(str(path/'*.mid*'))
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for n in range(args.num_generations):
        print(f"Generating sequence {n+1}...")
        random_file = random.choice(midi_files)
        seed_sequence = midi_to_notes(random_file)
        generated_notes = generate_sequence(rnn, seed_sequence, args.length, random_file)
        print(f"Generated sequence {n+1}")
        notes_to_midi(generated_notes, f"{args.output}/generated_{n+1}.mid", "Acoustic Grand Piano")
        