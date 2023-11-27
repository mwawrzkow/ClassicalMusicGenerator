import numpy as np
from midi import Midis

midis = Midis("/workspace/src/midi")
midis.read_all()
data = midis.get_notes() # this is a list of dictionaries, each dictionary is a note with its start and end times and velocity and channel and duration and note
print(len(data))



import os
import matplotlib as mpl 
mpl.use('WebAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

# print miltiple histograms in one plot for notes durations and velocities
notes = [x['note'] for x in data]
durations = [x['duration'] for x in data]
velocities = [x['velocity'] for x in data]
time = [x['time'] for x in data]
# now prepare plots
fig, axs = plt.subplots(1, 4, tight_layout=True)
axs[0].hist(notes, bins=128)
axs[0].set_title("Notes")
axs[1].hist(durations, bins=128)
axs[1].set_title("Durations")
axs[2].hist(velocities, bins=128)
axs[2].set_title("Velocities")
axs[3].hist(time, bins=128)
axs[3].set_title("Time")
# plt.show()
# now save the plot
fig.savefig("histograms.png")
# now we will start building the model cGAN
raw_data = np.array([[x['note'], x['duration'], x['velocity'], x['time']] for x in data])
notes = raw_data[:, 0]
durations = raw_data[:, 1]
velocities = raw_data[:, 2]
time = raw_data[:, 3]

# Normalize each feature separately
notes_normalized = (notes - notes.min()) / (notes.max() - notes.min())
durations_normalized = (durations - durations.min()) / (durations.max() - durations.min())
velocities_normalized = (velocities - velocities.min()) / (velocities.max() - velocities.min())
time_normalized = (time - time.min()) / (time.max() - time.min())

# Combine back into a single array
normalized_data = np.stack([notes_normalized, durations_normalized, velocities_normalized, time_normalized], axis=1)
batch_size = 2000
print(normalized_data.shape)
# trim the data to be a multiple of batch size
normalized_data = normalized_data[:-(normalized_data.shape[0] % batch_size)]
from network import GAN
# gan = GAN(1000, 3, 3)
gan = GAN(batch_size, 4, 4)
train_history = gan.train(normalized_data, 50_000 , batch_size)
# plot the training history it shows epoch and generator loss and discriminator loss
plt.plot(train_history[:, 0], label="Generator Sloss")
plt.plot(train_history[:, 1], label="Discriminator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_history.png")
# save the history to a file
np.save("training_history.npy", train_history)

# now generate some data
generated_data = gan.generate_data(batch_size)

def denormalize(data, original_data):
    generated_notes = data[:, 0]
    generated_durations = data[:, 1]
    generated_velocities = data[:, 2]
    generated_time = data[:, 3]

    generated_notes = generated_notes * (original_data[:, 0].max() - original_data[:, 0].min()) + original_data[:, 0].min()
    generated_durations = generated_durations * (original_data[:, 1].max() - original_data[:, 1].min()) + original_data[:, 1].min()
    generated_velocities = generated_velocities * (original_data[:, 2].max() - original_data[:, 2].min()) + original_data[:, 2].min()
    generated_time = generated_time * (original_data[:, 3].max() - original_data[:, 3].min()) + original_data[:, 3].min()
    generated_time = np.clip(generated_time, 0, None)
    return np.stack([generated_notes, generated_durations, generated_velocities, generated_time], axis=1)

# Ensure time values are non-negative
# now we have generated notes and durations and velocities
# now we need to convert them to midi
import mido
from mido import MidiFile, MidiTrack, Message

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
# now we need to convert the notes and durations and velocities to midi
generated = denormalize(generated_data, raw_data)
# now we need to convert the generated data to midi

generated_denoised_and_int = np.round(generated).astype(int)

# now we need to convert the generated data to midi
# at first we will create array of notes and durations and velocities as in midi file
# then we will sort them to be in order of time
# then we will create a midi file
midi_notes = []
for note, duration, velocity, time in generated_denoised_and_int:
    midi_notes.append({
        'type': 'note_on',
        'note': note,
        'time': time,
        'velocity': velocity,
        'channel': 0
    })
    midi_notes.append({
        'type': 'note_off',
        'note': note,
        'time': time + duration,
        'velocity': velocity,
        'channel': 0
    })
midi_notes = sorted(midi_notes, key=lambda x: x['time'])
# now create the midi file
for note in midi_notes:
    track.append(Message(note['type'], note=note['note'], velocity=note['velocity'], time=note['time'], channel=note['channel']))


    
# save the midi file
mid.save('output.mid')
