import numpy as np
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.from_tensor_slices_op import _TensorSliceDataset
from midi import Midis
import os
from network import GAN
import tensorflow as tf
# set TF_GPU_ALLOCATOR=cuda_malloc_async
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Clear the console
os.system('clear')
SKIP_NORMALIZATION = False # super parameter
data = None
skip = False
skipMidi = False
if not SKIP_NORMALIZATION:
    ASYNC = True


    if not __debug__:
        skip = bool(input("Do you want to load the model? y/N ").lower() == "y")
        skipMidi = bool(input("Do you want to skip reading the midis? y/N ").lower() == "y")

    if not skipMidi:
        midis = Midis("./Chopin")
        midis.read_all(100, ASYNC)
        data = midis.get_notes() 
        np.save("data.npy", data)
        print("Found {} tracks".format(len(data)))
        # exit()
    # save data to a file
    else:
        data = np.load("data.npy", allow_pickle=True)
        data = data.tolist()
        

    # remove midis from the memory 
    midis = None
    # this is a list of dictionaries, each dictionary is a note with its start and end times and velocity and channel and duration and note
    # ignore half of the notes
    data = data[:len(data)]




import os
import matplotlib as mpl 
mpl.use('WebAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

# print miltiple histograms in one plot for notes durations and velocities

# durations = [x['duration'] for x in data]
# now prepare plots for each feature
# notes_hist = plt.hist([x['note'] for x in data], bins=128)
# plt.xlabel("Note")
# plt.ylabel("Frequency")
# plt.savefig("notes_hist.png")
# plt.clf()
# # durations_hist = plt.hist(durations, bins=128)
# # plt.xlabel("Duration")
# # plt.ylabel("Frequency")
# # plt.savefig("durations_hist.png")
# # plt.clf()
# velocities_hist = plt.hist([x['velocity'] for x in data], bins=128)
# plt.xlabel("Velocity")
# plt.ylabel("Frequency")
# plt.savefig("velocities_hist.png")
# plt.clf()
# time_hist = plt.hist([x['time'] for x in data], bins=128)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.savefig("time_hist.png")
# plt.clf()
# now we will start building the model cGAN
# raw_data = np.array([[x['note'], x['duration'], x['velocity'], x['time']] for x in data])
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

notes = []
start_times = []
durations = []
velocities = []
channel = []

if not SKIP_NORMALIZATION:
    raw_data = data
    notes = [[y["note"] for y in x] for x in raw_data]
    start_times = [[y["start_time"] for y in x] for x in raw_data]
    durations = [[y["duration"] for y in x] for x in raw_data]
    velocities = [[y["velocity"] for y in x] for x in raw_data]
    channel =  [[y["channel"] for y in x] for x in raw_data]
# flatten the data
    notes = [item for sublist in notes for item in sublist]
    start_times = [item for sublist in start_times for item in sublist]
    durations = [item for sublist in durations for item in sublist]
    velocities = [item for sublist in velocities for item in sublist]
    channel = [item for sublist in channel for item in sublist]


    raw_data = None
    data = None
# Normalize each feature separately to be from -1 to 1

print(len(notes))




def normalize_list_of_lists(data):
    # Convert list of lists to a NumPy array
    array_data = np.array(data, dtype=float)  # Ensure data type is float for division

    # Check if the array is empty
    if array_data.size == 0:
        return array_data, (0, 0)  # Return empty array and dummy min/max
    
    # Compute the global min and max
    min_val = np.min(array_data)
    max_val = np.max(array_data)

    # Check for division by zero scenario
    if min_val == max_val:
        return np.zeros_like(array_data), (min_val, max_val)  # Return zero array and min/max
    
    # Normalize the data
    normalized_array = 2 * ((array_data - min_val) / (max_val - min_val)) - 1

    return normalized_array.tolist(), (min_val, max_val)

print("Normalizing data...")

# save data for future use as json file
notes_min_max = None
start_times_min_max = None
durations_min_max = None
velocities_min_max = None
channel_min_max = None
import json
if not SKIP_NORMALIZATION:
    print("Normalizing notes...")
    notes, notes_min_max = normalize_list_of_lists(notes)
    print("Normalizing start times...")
    start_times, start_times_min_max = normalize_list_of_lists(start_times)
    print("Normalizing durations...")
    durations, durations_min_max = normalize_list_of_lists(durations)
    print("Normalizing velocities...")
    velocities, velocities_min_max = normalize_list_of_lists(velocities)
    print("Normalizing channel...")
    channel, channel_min_max = normalize_list_of_lists(channel)
    print("Data normalized")
    print("Saving data...")
    # with open("min_max.json", "w") as f:
    #     json.dump({
    #         "notes": notes_min_max,
    #         "start_times": start_times_min_max,
    #         "durations": durations_min_max,
    #         "velocities": velocities_min_max,
    #         "channel": channel_min_max
    #     }, f)
    # print("Saved min/max values")
    # print(f"saving {len(notes)} notes")
    # with open("notes.json", "w") as f:
    #     json.dump(notes, f)
    # print("Saved notes")
    # print(f"saving {len(start_times)} start times")
    # with open("start_times.json", "w") as f:
    #     json.dump(start_times, f)
    # print("Saved start times")
    # print(f"saving {len(durations)} durations")
    # with open("durations.json", "w") as f:
    #     json.dump(durations, f,)
    # print("Saved durations")
    # print(f"saving {len(velocities)} velocities")
    # with open("velocities.json", "w") as f:
    #     json.dump(velocities, f)
    # print("Saved velocities")
    # print(f"saving {len(channel)} channel")
    # with open("channel.json", "w") as f:
    #     json.dump(channel, f)
    # print("Saved channel")
else: 
    with open("min_max.json") as f:
        min_max = json.load(f)
        notes_min_max = min_max["notes"]
        start_times_min_max = min_max["start_times"]
        durations_min_max = min_max["durations"]
        velocities_min_max = min_max["velocities"]
        channel_min_max = min_max["channel"]
    print("Loaded min/max values")
    with open("notes.json") as f:
        notes = json.load(f)
    print("Loaded notes")
    with open("start_times.json") as f:
        start_times = json.load(f)
    print("Loaded start times")
    with open("durations.json") as f:
        durations = json.load(f)
    print("Loaded durations")
    with open("velocities.json") as f:
        velocities = json.load(f)
    print("Loaded velocities")
    with open("channel.json") as f:
        channel = json.load(f)
    print("Loaded channel")

# Combine back into a single array
# ignore channel for a while channel
# normalized_data = np.stack([notes, start_times, durations, velocities], axis=1)
batch_size = 256

# dataset_notes = tf.data.Dataset.from_tensor_slices(notes)
# dataset_start_times = tf.data.Dataset.from_tensor_slices(start_times)
# dataset_durations = tf.data.Dataset.from_tensor_slices(durations)
# dataset_velocities = tf.data.Dataset.from_tensor_slices(velocities)

def parse_json(file_path):
    """Parses JSON files to extract data for each event."""
    raw_data = tf.io.read_file(file_path)
    
    def parse_json_fn(raw_data):
        data = json.loads(raw_data.numpy().decode('utf-8'))
        flat_data = [[float(item['note']), float(item['start_time']), float(item['duration']), float(item['velocity'])] for item in data]
        tensor = tf.convert_to_tensor(flat_data, dtype=tf.float32)
        # print("Tensor shape after conversion:", tensor.shape)
        return tensor
    
    result_tensor = tf.py_function(parse_json_fn, [raw_data], [tf.float32])[0]
    return result_tensor

def normalize(tensor):
    """Normalize the tensor."""
    tensor = tf.cast(tensor, tf.float32)
    mean = tf.reduce_mean(tensor, axis=0)
    std_dev = tf.math.reduce_std(tensor, axis=0)
    normalized_tensor = (tensor - mean) / (std_dev + 1e-10)
    return normalized_tensor

#batch_size = 256  # Adjust as needed
batch_size = 1024
dataset = tf.data.Dataset.list_files("./cache/*.json", shuffle=False)
dataset = dataset.map(parse_json)
dataset = dataset.unbatch()  # Unbatch the loaded data
dataset = dataset.shuffle(buffer_size=1000)  # Shuffle data
dataset = dataset.batch(batch_size, drop_remainder=True)  # Batch the data into the desired structure

# # Map the combined dataset to stack the tensors
# dataset = dataset.map(lambda notes, starts, durations, velocities: tf.stack([notes, starts, durations, velocities], axis=-1))


    # # Batch the dataset
# dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(buffer_size=10000)

print("Dataset created")
# for sample in dataset:
#     assert sample.shape == (batch_size, 4), "Invalid shape"

input_dim = (32,4)
from copy import deepcopy
# gan = GAN(1000, 3, 3)
generator_output_dim = (4,)  
discriminator_input_dim = deepcopy(generator_output_dim)
gan = GAN(input_dim, generator_output_dim, discriminator_input_dim, skip)
train_history = gan.train(dataset, 1000, 25)
# plot the training history it shows epoch and generator loss and discriminator loss
# if not skip:
#     plt.plot(train_history[:, 0], label="Generator Sloss")
#     plt.plot(train_history[:, 1], label="Discriminator loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig("training_history.png")
#     # save the history to a file
#     np.save("training_history.npy", train_history)

# now generate some data
generated_data = gan.generate_data(256)
# save generated data to a file
np.save("generated_data.npy", generated_data)

# def denormalize(data, original_data):
#     generated_notes = data[:, 0]
#     generated_durations = data[:, 1]
#     generated_velocities = data[:, 2]
#     generated_time = data[:, 3]

#     generated_notes = generated_notes * (original_data[:, 0].max() - original_data[:, 0].min()) + original_data[:, 0].min()
#     generated_durations = generated_durations * (original_data[:, 1].max() - original_data[:, 1].min()) + original_data[:, 1].min()
#     generated_velocities = generated_velocities * (original_data[:, 2].max() - original_data[:, 2].min()) + original_data[:, 2].min()
#     generated_time = generated_time * (original_data[:, 3].max() - original_data[:, 3].min()) + original_data[:, 3].min()
#     generated_time = np.clip(generated_time, 0, None)
#     return np.stack([generated_notes, generated_durations, generated_velocities, generated_time], axis=1)
def denormalize_list_of_lists(normalized_data, min_max):
    # Extract the min and max values used for normalization
    min_val, max_val = min_max
    
    # Convert the normalized list of lists to a NumPy array
    normalized_array = np.array(normalized_data, dtype=float)
    
    # Check if the array is empty or if min_val equals max_val
    if normalized_array.size == 0 or min_val == max_val:
        return normalized_array.tolist()  # Return the original array as list
    
    # Denormalize the data
    denormalized_array = ((normalized_array + 1) / 2) * (max_val - min_val) + min_val
    
    return denormalized_array.tolist()

# Ensure time values are non-negative
# now we have generated notes and durations and velocities
# now we need to convert them to midi
import mido
from mido import MidiFile, MidiTrack, Message

mid = MidiFile()
# set default values for midi file
mid.ticks_per_beat = 480

track = MidiTrack()
mid.tracks.append(track)
# denormalize generated data
generated = denormalize_list_of_lists(generated_data, notes_min_max)
# remove last output.json if exists
if os.path.exists("output.json"):
    os.remove("output.json")
open("output.json", "w").write(json.dumps(generated))


# now we need to convert the generated data to midi

generated_denoised_and_int = np.round(generated).astype(int)

# now we need to convert the generated data to midi
# at first we will create array of notes and durations and velocities as in midi file
# then we will sort them to be in order of time
# then we will create a midi file
midi_notes = []
# for note, duration, velocity, time in generated_denoised_and_int:
for note, start_time, duration, velocity in generated_denoised_and_int:
    midi_notes.append({
        'type': 'note_on',
        'note': note,
        'time': start_time,
        'duration': duration,
        'velocity': velocity,
        'channel': 0
    })

# convert start time and duration to note off
notes_off = []
for note in midi_notes:
    notes_off.append({
        'type': 'note_off',
        'note': note['note'],
        'time': note['time'] + note['duration'],
        'velocity': note['velocity'],
        'channel': 0
    })

midi_notes.extend(notes_off)
# sort the notes by time
midi_notes = sorted(midi_notes, key=lambda x: x['time'])

    # midi_notes.append({
    #     'type': 'note_off',
    #     'note': note,
    #     'time': time + duration,
    #     'velocity': velocity,
    #     'channel': 0
    # })
# midi_notes = sorted(midi_notes, key=lambda x: x['time'])
# now create the midi file from the notes with respective durations and velocities
# now we need to convert the generated data to midi
for note in midi_notes:
    if note['type'] == 'note_on':
        track.append(Message('note_on', note=note['note'], velocity=note['velocity'], time=int(note['time'])))
    else:
        track.append(Message('note_off', note=note['note'], velocity=note['velocity'], time=int(note['time'])))



    
# save the midi file
mid.save('output.mid')
mid.play()

# Convert orginal data to midi
