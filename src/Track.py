import mido
from mido import MidiFile, Message

class Track:
    def __init__(self, midi_file, track_id=0):
        self.midi_file = midi_file
        self.track_id = track_id
        self.notes = []
        self.ticks_per_beat = midi_file.ticks_per_beat
        self.tempo = 500000  # Default MIDI tempo (500,000 microseconds per beat)

    def read_notes(self):
        track = self.midi_file.tracks[self.track_id]
        time_elapsed = 0
        current_notes = {}

        for msg in track:
            if msg.type == 'set_tempo':
                self.tempo = msg.tempo
            if msg.time > 0:
                time_elapsed += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                note_info = {
                    'note': msg.note,
                    'start_time': self._convert_to_quarter_length(time_elapsed),
                    'velocity': msg.velocity,
                    'channel': msg.channel
                }
                current_notes[msg.note] = note_info

            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in current_notes:
                start_time = current_notes[msg.note]['start_time']
                duration = self._convert_to_quarter_length(time_elapsed) - start_time
                self.notes.append({
                    'note': msg.note,
                    'start_time': start_time,
                    'duration': duration,
                    'velocity': current_notes[msg.note]['velocity'],
                    'channel': current_notes[msg.note]['channel']
                })
                del current_notes[msg.note]

    def _convert_to_quarter_length(self, ticks):
        quarter_length = (ticks / self.ticks_per_beat) * (self.tempo / 1000000.0)
        return quarter_length

    def get_notes(self):
        return self.notes
