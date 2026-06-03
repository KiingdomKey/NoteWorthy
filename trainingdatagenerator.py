from constants import *

import numpy as np
import random
from types import SimpleNamespace
import os

import librosa
import soundfile
import pretty_midi
import IPython
import fluidsynth

import numpy as np
import random
from types import SimpleNamespace
import os

import librosa
import soundfile
import pretty_midi
import IPython
import fluidsynth

class TrainingDataGenerator:
    
    def __init__(self,
                 audio_output_path: str,
                 label_output_path: str,
                 default_instrument_name: str = "Acoustic Grand Piano"):

        self.audio_output_path = audio_output_path
        self.label_output_path = label_output_path

        if default_instrument_name not in pretty_midi.constants.INSTRUMENT_MAP:
            default_instrument_name = "Acoustic Grand Piano"
            print("Instrument not in pretty_midi. Defaulting to piano.")
        self.default_instrument_name = default_instrument_name

        valid_notes = []
        valid_notes += ['A0', 'Bb0', 'B0']
        for i in range(1,8):
            valid_notes += [f'C{i}',
                            f'Db{i}',
                            f'D{i}',
                            f'Eb{i}',
                            f'E{i}',
                            f'F{i}',
                            f'Gb{i}',
                            f'G{i}',
                            f'Ab{i}',
                            f'A{i}', 
                            f'Bb{i}',
                            f'B{i}']
        valid_notes += ['C8']
        self.valid_notes = valid_notes

    def _is_note_overlapping(self,
                         note: pretty_midi.Note,
                         instrument: pretty_midi.Instrument) -> bool:

        
        for instrument_note in instrument.notes:
            instrument_note_start = instrument_note.start - MIN_NOTE_INTERVAL
            instrument_note_end = instrument_note.end + MIN_NOTE_INTERVAL
            if (note.pitch == instrument_note.pitch) \
                and ((instrument_note_start <= note.start <= instrument_note_end) or (instrument_note_start <= note.end <= instrument_note_end)):
                    return True 

        return False

    # Note: pretty_midi uses seconds for time (i.e., 1.0 = 1 second)
    def generate_midi(self, 
                      num_notes: int, 
                      max_total_length: float,
                      min_note_length: float,
                      max_note_length: float,
                      velocity_range: tuple[int, int] = (20, 100),) -> pretty_midi.Instrument:

        instrument_program = pretty_midi.instrument_name_to_program(self.default_instrument_name)
        instrument = pretty_midi.Instrument(program=instrument_program)

        for _ in range(0, num_notes):

            note_overlapping = True
            while note_overlapping:
                random_note = self.valid_notes[random.randint(0, len(self.valid_notes) - 1)]

                note_number = pretty_midi.note_name_to_number(random_note)

                start = random.uniform(0, max_total_length - MIN_NOTE_INTERVAL)
                end = np.min([start + random.uniform(min_note_length, max_note_length), max_total_length])

                velocity = random.randint(*velocity_range)

                note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=start, end=end)
                
                note_overlapping = self._is_note_overlapping(note, instrument)

            instrument.notes.append(note)

        return instrument

    def midi_to_audio(self,
                      instrument: pretty_midi.Instrument,
                      synthesizer: str = None,
                      use_clean_audio=True) -> np.array:

        audio_data = instrument.fluidsynth(synthesizer=synthesizer)

        if use_clean_audio:
            clean_audio_obj = IPython.display.Audio(audio_data, rate=SAMPLE_RATE)
            audio_data = np.frombuffer(clean_audio_obj.data, dtype=np.int16)

        return audio_data

    def output_audio(self,
                     audio_data: np.array,
                     file_name: str):

        current_dir = os.getcwd()
        audio_subdir = os.path.join(self.audio_output_path)

        os.makedirs(os.path.join(current_dir, audio_subdir), exist_ok=True)

        file_path = os.path.join(current_dir, audio_subdir, file_name)

        with open(file_path, "wb") as f:
            f.write(audio_data)  

    def midi_to_label(self,
                      instrument: pretty_midi.Instrument,
                      audio_length: int) -> np.array:
        
        time_steps = int(audio_length / (CQT_STEP_SIZE* TIME_STEP_LENGTH)) + 1
        labels = np.zeros((CQT_NUM_BUCKETS, time_steps))

        for note in instrument.notes:

            start = int(np.round(note.start * SAMPLE_RATE / (CQT_STEP_SIZE * TIME_STEP_LENGTH)))
            end = int(np.round(note.end * SAMPLE_RATE / (CQT_STEP_SIZE * TIME_STEP_LENGTH)))

            pitch = note.pitch - LOWEST_NOTE_OFFSET
            labels[pitch, start] = NOTE_ONSET_SYMBOL
            labels[pitch, start + 1:end] = NOTE_CONTINUED_SYMBOL
            labels[pitch, end] = NOTE_END_SYMBOL

        return labels

    def output_labels(self,
                     labels: np.array,
                     file_name: str):

        current_dir = os.getcwd()
        label_subdir = os.path.join(self.label_output_path)

        os.makedirs(os.path.join(current_dir, label_subdir), exist_ok=True)

        file_path = os.path.join(current_dir, label_subdir, file_name)

        np.save(file_path, labels)


    def generate_and_output_audio_and_label(self,
                                        num_notes: int,
                                        max_total_length: float,
                                        min_note_length: float,
                                        max_note_length: float,
                                        root_file_name: str,
                                        velocity_range: tuple[int, int] = None) -> None:
        
        instrument = self.generate_midi(num_notes=num_notes,
                                        max_total_length=max_total_length,
                                        min_note_length=min_note_length,
                                        max_note_length=max_note_length)
        
        audio_data = self.midi_to_audio(instrument=instrument)
        self.output_audio(audio_data=audio_data, file_name=root_file_name + ".mp3")

        labels = self.midi_to_label(instrument=instrument, audio_length=len(audio_data))

        self.output_labels(labels=labels, file_name=root_file_name + ".npy")