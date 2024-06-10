import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import os
import concurrent.futures
from pathlib import Path
# i want to check how much time it takes to read all the midis
import time
import json # To save the notes dictionary to a json file
from tqdm import tqdm
from Track import Track

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
        filename = str(self.deterministic_hash(midi[0])) + ".json"
        data = []
        if os.path.exists(os.path.join("cache", filename)):
            with open(os.path.join("cache", filename), "r") as f:
                data = json.load(f)                
        else:
            try:
                mid = MidiFile(midi[1])
                for i, track in enumerate(mid.tracks):
                    track = Track(mid, i)
                    track.read_notes()
                    notes.extend(track.get_notes())
                # get ony 512 notes
                data = notes
                self.save_notes(notes, filename)
            except Exception as e:
                print(f"Error processing {midi[0]}: {e}") 
        # find data where channel is not 0 and print it
        non_zero_channel = [note for note in data if note['channel'] != 0]
        if non_zero_channel:
            print(f"Non-zero channel in {midi[0]}: {non_zero_channel}")

        

        return data
        
    def save_notes(self, notes, filename):
        if not os.path.exists("cache"):
            os.mkdir("cache")
        path = os.path.join("cache", filename)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(notes, f)
    
    def deterministic_hash(self, to_hash) -> str:
        # return deterministic hash of the string
        hashed = 0
        for i, char in enumerate(to_hash):
            hashed += ord(char) * (i + 1)
        return str(hashed)

    def read_all(self, percentage=100, ASYNC=True):
        start_time = time.time()
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".mid"):
                    self.files.append((file, os.path.join(root, file)))
        data = []
        # do smth like htis:
        self.files = self.files[:int(len(self.files) * percentage / 100)]
        # make sorted list of all midis
        sorted_midis = sorted(self.files, key=lambda x: x[0])
        midi_combined = "".join([midi[0] for midi in sorted_midis])
        midi_hashed = str(self.deterministic_hash(midi_combined))
            # with concurrent.futures.ThreadPoolExecutor() as executor:
        if ASYNC:
            with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                
                executors = [executor.submit(self.read_midi, midi) for midi in self.files]
                for future in tqdm(concurrent.futures.as_completed(executors), total=len(sorted_midis), desc="Processing MIDI files"):
                    # clear console
                    try: 
                        data.append(future.result())
                    except Exception as e:
                        print(e)
                        continue
        else:
            for midi in self.files:
                data.append(self.read_midi(midi))


        print(f"Reading all midis took {time.time() - start_time} seconds")
            
        # start_time = time.time()
        # self.notes = self.append_duration_to_notes(data)
        # trim all tracks to have the same length
        self.notes = self.trim_tracks(data)
        # check if all tracks have the same length
        for i, track in enumerate(self.notes):
            if len(track) != len(self.notes[0]):
                print(f"Track {i} has a different length: {len(track)}")

        # self.notes = data
    def trim_tracks(self, notes):
        # max_length = max([len(track) for track in notes])
        max_length = 512
        for track in notes:
            if len(track) < max_length:
                track.extend([{'note': 0, 'start_time': 0, 'duration': 0, 'velocity': 0, 'channel': 0}] * (max_length - len(track)))
            elif len(track) > max_length:
                new_track = track[:max_length]
                notes[notes.index(track)] = new_track
        return notes
        
    def get_notes(self) -> list[dict]:
        return self.notes