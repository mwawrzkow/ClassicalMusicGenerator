import pretty_midi

class Track:
    def __init__(self, midi_data, track_id=0):
        self.midi_data = midi_data
        self.track_id = track_id
        self.notes = []
        self.ticks_per_beat = midi_data.resolution
        self.tempo = 500000  # Default MIDI tempo (500,000 microseconds per beat)
        self._initialize_tempo()

    def _initialize_tempo(self):
        if self.midi_data.get_tempo_changes()[1].size > 0:
            self.tempo = self.midi_data.get_tempo_changes()[1][0]  # Get the first tempo change

    def read_notes(self):
        instrument = self.midi_data.instruments[self.track_id]
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = float(sorted_notes[0].start)
        for note in instrument.notes:
            start = float(note.start)
            end = float(note.end)
            proposal = {
                'start': start,
                'end': end, 
                'pitch': int(note.pitch),
                'step': start-prev_start,
                'duration': end-start,
            }
            self.notes.append(proposal)
            prev_start = start

    def _convert_to_quarter_length(self, time_in_seconds):
        quarter_length = (time_in_seconds * 1e6) / self.tempo
        return quarter_length

    def get_notes(self):
        return self.notes
