from constants import *

import os
import numpy as np
from types import SimpleNamespace

import librosa

import os
import numpy as np
from types import SimpleNamespace

import librosa

class DataPreparer:

    def __init__(self,
                 audio_file_path: str,
                 prepared_data_output_path: str,
                 sample_rate: int = 22050,
                 cqt_step_size: int = 512):

        self.audio_file_path = audio_file_path
        self.prepared_data_output_path = prepared_data_output_path
        self.sample_rate = sample_rate
        self.cqt_step_size = cqt_step_size


    def audio_file_to_cqt(self,
                          file_name) -> (np.array, int):

        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, self.audio_file_path, file_name)

        y, sr = librosa.load(file_path)
        C = np.abs(librosa.cqt(y, sr=self.sample_rate, bins_per_octave=12, n_bins=CQT_NUM_BUCKETS, hop_length=self.cqt_step_size))
        return C, sr
        
    def output_cqt_file(self,
                        C: np.array,
                        file_name: str) -> None:

        current_dir = os.getcwd()
        os.makedirs(os.path.join(current_dir, self.prepared_data_output_path), exist_ok=True)

        file_path = os.path.join(current_dir, self.prepared_data_output_path, file_name)
        
        np.save(file_path, C)

    def prepare_audio(self,
                      root_file_name: str) -> None:
        
        C, sr = self.audio_file_to_cqt(file_name=root_file_name + ".mp3")
        self.output_cqt_file(C=C, file_name=root_file_name + ".npy")