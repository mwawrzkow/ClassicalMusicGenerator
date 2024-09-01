import collections
import glob
import pathlib
from typing import Optional
import numpy as np
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.from_tensor_slices_op import _TensorSliceDataset
from midi import Midis
import os
from network_gan import GAN
from network_rrn import RNNModel
import tensorflow as tf
import pandas as pd 
import pretty_midi
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Clear the console
os.system('clear')
SKIP_NORMALIZATION = False # super parameter
data = None
skip = False
skipMidi = False
ticks_per_beat = 0
if not SKIP_NORMALIZATION:
    ASYNC = True


    if not __debug__:
        skip = bool(input("Do you want to load the model? y/N ").lower() == "y")
        skipMidi = bool(input("Do you want to skip reading the midis? y/N ").lower() == "y")

    if not skipMidi:
        midis = Midis("./Chopin")
        midis.read_all(100, ASYNC)
        data = midis.get_notes() 
        ticks_per_beat = midis.get_avarage_ticks_per_beat()
        np.save("data.npy", data)
        print("Found {} tracks".format(len(data)))
    else:
        data = np.load("data.npy", allow_pickle=True)
        data = data.tolist()
        
    midis = None
    data = data[:len(data)]




import os
import matplotlib as mpl 
mpl.use('WebAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Bidirectional, BatchNormalization, Reshape, Input
from tqdm import tqdm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)


import json

notes = None
start_times = None
durations = None
velocities = None
channel = None


batch_size = 64
sequence_length = 512

def parse_json(file_path):
    """Parses JSON files to extract data for each event."""
    raw_data = tf.io.read_file(file_path)
    
    def parse_json_fn(raw_data):
        data, bpm = json.loads(raw_data.numpy().decode('utf-8'))
        data = data[:batch_size]
        flat_data = [[float(item['note']),
              float(item['duration']),
              float(item['velocity'])] for item in data]
        tensor = tf.convert_to_tensor(flat_data, dtype=tf.float32)
        return tensor
    
    result_tensor = tf.py_function(parse_json_fn, [raw_data], [tf.float32])[0]
    result_tensor.set_shape([None, 3])  # Set shape to (None, 4) as we expect 4 features per timestep
    return result_tensor

print("Dataset created")

CACHE_PATH = "./cache"
json_files = [f for f in os.listdir(CACHE_PATH) if f.endswith('.json')]
            
            
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
data_dir = pathlib.Path("midi")
# filenames = glob.glob(str(data_dir/'**/*.mid*'))
filenames = glob.glob(str(data_dir/'*.mid*'))

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)
#   save the plot to a file
  plt.savefig("piano_roll.png")

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

print(filenames[0])
plot_piano_roll(midi_to_notes(filenames[0]))
notes_to_midi(midi_to_notes(filenames[0]), "output.mid", "Acoustic Grand Piano")
# now we have all the data in the data variable
# drop time and channel
def json_to_dataframe(json_files):
    data = [] 
    for file in json_files:
        with open(os.path.join(CACHE_PATH, file)) as f:
            file_data, bpm = json.load(f)
            notes = collections.defaultdict(list)
            for item in file_data:
                notes['pitch'].append(item['pitch'])
                notes['duration'].append(item['duration'])
                notes['step'].append(item['step'])
            data.append(pd.DataFrame({name: np.array(value) for name, value in notes.items()}))
    return data
# dataframe = json_to_dataframe(json_files)


num_files = 1000
all_notes = []
import concurrent.futures

with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
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
key_order = ['pitch', 'step', 'duration']
data = np.stack([all_notes[key] for key in key_order], axis=1)


# data = [[x['note'], x['duration'], x['velocity']] for x in data]
# combine 
# ensure each array in data is normalized before anything: 
data = np.array(data)
dataset = tf.data.Dataset.from_tensor_slices(data)


# Function to generate sequences and targets
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


# Generate input sequences and targets
tf_dataset = create_sequences_tf(dataset, sequence_length)

# If the input length is less than sequence_length, pad the sequences
# Convert to TensorFlow datasets
# dataset = 

# Shuffle and batch the dataset
dataset = tf_dataset.shuffle(buffer_size=len(data) // sequence_length).batch(batch_size, drop_remainder=True)

# Print the dataset shapes
for input_batch, target_batch in dataset.take(1):
    print("Input batch shape:", input_batch.shape)
    print("Target batch shape:", target_batch) # Dictionary of labels
    # print first 5 elements of first batch 
    print("First 5 elements of first batch:")
    print(input_batch[0][:5])


input_dim = (128,4)
from copy import deepcopy
# gan = GAN(1000, 3, 3)
generator_output_dim = (3,)
output_dim = (4,)  
from network_rrn import RNNModel
discriminator_input_dim = deepcopy(generator_output_dim)
generator_generated = None
# gan = GAN(input_dim, generator_output_dim, discriminator_input_dim, skip)
# generator_generated = gan.train(dataset, 1, 2)
# for batch in dataset.take(1):
#     print("Batch shape:", batch.shape)
#     # save one batch to batch.json
#     with open("batch.json", "w") as f:
#         json.dump(batch.numpy().tolist(), f)
rnn = RNNModel((sequence_length,3), 4)
# rnn.load_checkpoint("training_checkpoints/ckpt_4")
rnn.train(dataset, 500)

import random
filename = random.sample(filenames, 1)[0]
raw_notes = midi_to_notes(filename)

temperature = 3.0
num_predictions = 1000

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
input_notes = (
    sample_notes[:sequence_length] / np.array([128, 1, 1]))

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
  pitch, step, duration = rnn.predict_next_note(input_notes,temperature)
  start = prev_start + step
  end = start + duration
  input_note = (pitch, step, duration)
  generated_notes.append((*input_note, start, end))
  input_notes = np.delete(input_notes, 0, axis=0)
  input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
  prev_start = start

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))
notes_to_midi(generated_notes, "generated.mid", "Acoustic Grand Piano")
gan = rnn 
def convert_to_midis(data, filename):
    # check if training_midi directory exists
    if not os.path.exists("training_midi"):
        os.mkdir("training_midi")
    # modify filename to be in the training_midi directory
    filename = os.path.join("training_midi", filename)
    np.save(filename + ".npy", data)
    generated_data_from_file = np.load(filename+ ".npy")
    df = pd.DataFrame(generated_data_from_file)
    df.to_csv(filename +".csv", index=False)
    def denormalize_list_of_lists(normalized_data, min_max_notes, min_max_start_times, min_max_durations, min_max_velocities):
        # Extract the min and max values used for normalization
        notes_min, notes_max = min_max_notes
        start_times_min, start_times_max = min_max_start_times
        durations_min, durations_max = min_max_durations
        velocities_min, velocities_max = min_max_velocities
        
        # Convert the normalized list of lists to a NumPy array
        normalized_array = np.array(normalized_data, dtype=float)
        
        # Check if the array is empty or if min_val equals max_val
        if normalized_array.size == 0 or notes_min == notes_max or start_times_min == start_times_max or durations_min == durations_max or velocities_min == velocities_max:
            return normalized_array.tolist()  # Return the original array as list
        
        # Denormalize the data
        denormalized_array = np.zeros_like(normalized_array)
        # denormalized_array[:, 0] = (normalized_array[:, 0] + 1) / 2 * (notes_max - notes_min) + notes_min
        # denormalized_array[:, 1] = (normalized_array[:, 1] + 1) / 2 * (start_times_max - start_times_min) + start_times_min
        denormalized_array[:, 1] = (normalized_array[:, 1] + 1) / 2 * (durations_max - durations_min) + durations_min
        denormalized_array[:, 2] = (normalized_array[:, 2] + 1) / 2 * (velocities_max - velocities_min) + velocities_min
        # ensure all values int and non-negative
        denormalized_array = np.round(denormalized_array).astype(int)
        denormalized_array = np.clip(denormalized_array, 0, None)
        # remove all notes with duration 0 or start time 0
        denormalized_array = denormalized_array[denormalized_array[:, 1] > 0]
        denormalized_array = denormalized_array[denormalized_array[:, 2] > 0]

        
        return denormalized_array.tolist()

    from mido import MidiFile, MidiTrack, Message
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(
    program=pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano'))

    

    # denormalize generated data
    generated = data.tolist() #denormalize_list_of_lists(data, notes_min_max, start_times_min_max, durations_min_max, velocities_min_max)
    if os.path.exists(filename+ ".json"):
        os.remove(filename+ ".json")
    open(filename+".json", "w").write(json.dumps(generated))

    # create a midi file from the generated data
    prev_start = 0 
    for _ in generated: 
        pitch, step, duration = _
        pitch = int(pitch)
        step = int(step)
        duration = int(duration) 
        start = prev_start + step
        end = start + duration
        prev_start = start
        instrument.notes.append(pretty_midi.Note(
            velocity=100, pitch=pitch, start=start, end=end
        ))
    pm.instruments.append(instrument)
    pm.write(filename + ".mid")
os.system("rm -rf training_midi")
