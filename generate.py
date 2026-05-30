from trainingdatagenerator import TrainingDataGenerator
from datapreparer import DataPreparer

import os
import sys
import argparse
import random
import tqdm

AUDIO_DIR = "data/audio"
LABEL_DIR =  "data/labels"
PREPARED_DATA_DIR = "data/prepared_data"
PRINT_EVERY_N_FILES = 1

parser = argparse.ArgumentParser()
parser.add_argument("num_files", help="Number of Files to Generate")
parser.add_argument("-mnnn", "--min_num_notes", help="Minimum Number of Notes Per File")
parser.add_argument("-mxnn", "--max_num_notes", help="Maximum Number of Notes Per File")
parser.add_argument("-mnl", "--min_length", help="Max Length of a File in Seconds")
parser.add_argument("-mxl", "--max_length", help="Max Length of a File in Seconds")
parser.add_argument("-mnnl", "--min_note_length", help="Minimum Length for a Note in Seconds")
parser.add_argument("-mxnl", "--max_note_length", help="Maximum Length for a Note in Seconds")

args = parser.parse_args()
num_files_to_make = int(args.num_files) if args.num_files is not None else 128
min_num_notes = int(args.min_num_notes if args.min_num_notes is not None else 100)
max_num_notes = int(args.max_num_notes if args.max_num_notes is not None else 200)
min_length = float(args.min_length) if args.min_length is not None else 10
max_length = float(args.max_length) if args.max_length is not None else 60
min_note_length = float(args.min_note_length) if args.min_note_length is not None else 0.5
max_note_length = float(args.max_note_length) if args.max_note_length is not None else 5

# If any of the minimums are greater than the maximums then we just swap them to avoid any errors
if min_num_notes > max_num_notes:
    min_num_notes, max_num_notes = max_num_notes, min_num_notes
if min_length > max_length:
    min_length, max_length = max_length, min_length
if min_note_length > max_note_length:
    min_note_length, max_note_length = max_note_length, min_note_length

generator = TrainingDataGenerator(audio_output_path=AUDIO_DIR,
                                  label_output_path=LABEL_DIR)

preparer = DataPreparer(audio_file_path=AUDIO_DIR,
                        prepared_data_output_path=PREPARED_DATA_DIR)

for i in tqdm.tqdm(range(num_files_to_make),
                   desc="Generating Files",
                   unit=" Files"):

    root_file_name = f"{i:04d}"

    generator.generate_and_output_audio_and_label(num_notes=random.randint(min_num_notes, max_num_notes),
                               max_total_length=random.uniform(min_length, max_length),
                               min_note_length=min_note_length,
                               max_note_length=max_note_length,
                               root_file_name=root_file_name)

    preparer.prepare_audio(root_file_name=root_file_name)

print("\nDone!")

