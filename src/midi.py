import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import os
import concurrent.futures
from pathlib import Path
# i want to check how much time it takes to read all the midis
import time
import json # To save the notes dictionary to a json file


class Midis:
    def __init__(self, path):
        self.path = path
        self.files = []
        self.notes = []  # Dictionary to store notes with their start and end times

    def read_midi(self, midi):
        notes = []
        # at first check if the midi file has already been read and saved
        # if it has been read and saved, then just read the json file
        # if not, then read the midi file and save it
        filename = midi[0].replace(".mid", ".json")
        data = []
        if os.path.exists(os.path.join("cache", filename)):
            with open(os.path.join("cache", filename), "r") as f:
                data = json.load(f)                
        else:
            mid = MidiFile(midi[1])
            for i, track in enumerate(mid.tracks):
                for msg in track:
                    if msg.type == 'note_on' or msg.type == 'note_off':
                        data.append({
                            'type': msg.type,
                            'note': msg.note,
                            'time': msg.time,
                            'velocity': msg.velocity,
                            'channel': msg.channel
                        })
        return data
        
    def save_notes(self, notes):
        # check if cache folder exists
        if not os.path.exists("cache"):
            os.mkdir("cache")
        # save notes to a json file
        for filename, data in notes:
            filename = filename.replace(".mid", ".json")
            # check if the file already exists
            if not os.path.exists(os.path.join("cache", filename)):
                with open(os.path.join("cache", filename), "w") as f:
                    json.dump(data, f)

    def append_duration(self, notes):
        # calculate each note duration
        # do that in a separate threads
        # this will speed up the process
        for i, note in enumerate(notes):
            if note['type'] == 'note_on':
                for j, n in enumerate(notes[i+1:]):
                    if n['type'] == 'note_off' and n['note'] == note['note']:
                        note['duration'] = n['time']
                        break
        # remove field type from notes
        notes_without_type = [ {k: v for k, v in note.items() if k != 'type'} for note in notes]
        return [note for note in notes_without_type if 'duration' in note]
    
    def append_duration_to_notes(self, notes):
        # calculate each note duration
        # do that in a separate threads
        # this will speed up the process
        data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor and collect futures
            future_to_filename = {executor.submit(self.append_duration, notes): filename for filename, notes in notes}

            # Retrieve and store results as they complete
            for future in concurrent.futures.as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    duration = future.result()
                except Exception as exc:
                    print(f'{filename} generated an exception: {exc}')
                else:
                    data.append((filename, duration))
        return data

    def read_all(self):
        start_time = time.time()
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".mid"):
                    self.files.append((file, os.path.join(root, file)))
        data = []
        for midi in self.files:
            data.append((midi[0],self.read_midi(midi)))
        self.save_notes(data)
        print(f"Reading all midis took {time.time() - start_time} seconds")
        start_time = time.time()
        data = self.append_duration_to_notes(data)
        [self.notes.extend(notes) for filename, notes in data]
        print(f"Calculating durations took {time.time() - start_time} seconds")
    def get_notes(self):
        return self.notes