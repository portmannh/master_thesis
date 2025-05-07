# %%
"""
Preparing SC data as:
- paths for whole night testing  (pretext task)
- 5.5 hour tf.data.Datasets for training, validation, and 17.5 minute pre-testing (pretext task)
- including age information for age regression in downstream task
"""

# %%
# imports
import mne
import numpy as np
import os
import tensorflow as tf
import re
from collections import defaultdict
import json
import pandas as pd
import random

# set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# %%
# FUNCTIONS TO RETRIEVE AND PREPARE DATA

# %%
# Load age data from excel
def load_ages(age_file_path):
    descriptions = pd.read_excel(age_file_path)
    age_dict = {row['subject']: row['age'] for _, row in descriptions.iterrows()}
    return age_dict

# get lists of psg file paths and corresponding hypnogram file paths as well as corresponding ages
def get_data_paths(directory, age_dict):
    all_files = os.listdir(directory)

    psg_files = [f for f in all_files if f.endswith('PSG.edf')]
    hypnogram_files = [f for f in all_files if f.endswith('Hypnogram.edf')]
    
    sorted_psg_paths = []
    sorted_hypnogram_paths = []
    ages = []

    # pair psg and hypnogram files and age based on participant and night name -> there is always a pair of PSG and Hypnogram files
    # e.g. for SC4001E0-PSG.edf extract identifier SC4001
    for psg_file in psg_files:
        base_name = psg_file.split('-')[0]
        identifier = base_name[:-2]
        subject_id = extract_subject_id(psg_file)
        for hypnogram_file in hypnogram_files:
            if hypnogram_file.startswith(identifier):
                sorted_psg_paths.append(os.path.join(directory, psg_file))
                sorted_hypnogram_paths.append(os.path.join(directory, hypnogram_file))
                ages.append(age_dict.get(subject_id, -1))

    # returns 2 lists of paths (1 for PSG 1 for Hypnograms) and one list of ages sorted in the same order
    return sorted_psg_paths, sorted_hypnogram_paths, ages


# Function to extract subject ID as an integer from the edf file path
# e.g. for SC4001E0-PSG.edf extract 0 (participant 0)
def extract_subject_id(file_path):
    # extract subject number from file name
    pattern = r'SC4(\d{2})\d'
    match = re.search(pattern, file_path)
    if match:
        return int(match.group(1)) # return subject ID as integer (0 would be intire matched string)
    return None


# Function to flatten list of files for each subject
# ignores the subject number that was used to sort the files
def flatten_subject_files(subject_list):
    return [file_tuple for subject, files in subject_list for file_tuple in files]


# Function to split data based on subject ID
def split_dataset_by_subjects(psg_paths, hypnogram_paths, ages, train_ratio=0.7, val_ratio=0.15):
    # create a dictionary to group files by subject ID (this creates per default an empty list per key)
    subject_files = defaultdict(list)

    for psg_path, hypnogram_path, age in zip(psg_paths, hypnogram_paths, ages):
        subject_id = extract_subject_id(psg_path)

        # append file and age tuple to corresponding subject
        subject_files[subject_id].append((psg_path, hypnogram_path, age))

    # convert dictionary to list of (subject_id, [(file_paths)]) and shuffle
    subject_list = list(subject_files.items())
    np.random.shuffle(subject_list)

    # calculate number of subjects for each split
    num_subjects = len(subject_list)
    train_end = int(train_ratio * num_subjects)
    val_end = train_end + int(val_ratio * num_subjects)

    # split subject list into train, val, and test
    train_subjects = subject_list[:train_end]
    val_subjects = subject_list[train_end:val_end]
    test_subjects = subject_list[val_end:]

    # convert list of subjects and file tuples to list of file tuples (now not in lists ordered by subject anymore)
    train_files = flatten_subject_files(train_subjects)
    val_files = flatten_subject_files(val_subjects)
    test_files = flatten_subject_files(test_subjects)

    return train_files, val_files, test_files


# convert sleep stage (string) to numeric stage
# group S3 and S4 together (AASM standard)
def map_sleep_stage_to_label(stage):
    stage_mapping = {
    'Sleep stage W': 0,
    'Sleep stage R': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage 4': 4
    }
    return stage_mapping.get(stage, -1) # -1 is Sleep Stage ? or Movement time


# load data from edf files
def load_data(psg_path, hypnogram_path):
    
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose='warning')

    # choose channels I want to use
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    raw.pick_channels(channels)

    psg_data = raw.get_data()

    hypnogram = mne.read_annotations(hypnogram_path)

    n_samples = len(raw.times)
    sample_rate = raw.info['sfreq']

    sleep_stage_labels = np.zeros(n_samples, dtype=np.int32)

    # create list containing sleep stage for every time point
    for row in hypnogram:
        onset_sample = int(row['onset'] * sample_rate)
        duration_sample = int(row['duration'] * sample_rate)
        stage = row['description']

        stage_label = map_sleep_stage_to_label(stage)

        sleep_stage_labels[onset_sample:onset_sample + duration_sample] = stage_label

    return (psg_data, sleep_stage_labels)


# removing wake stages more than 30 min before and after sleep
def remove_wake(signals, labels, wake_stage_index=0, sampling_rate=100):

    thirty_min_samples = 30 * 60 * sampling_rate

    # find indices of all non-wake stages
    non_wake_indices = np.where(labels != wake_stage_index)[0]

    if len(non_wake_indices) == 0:
        raise ValueError("No non-wake stages found in the data.")
    
    # get first and last non-wake stage indices
    first_non_wake_index = non_wake_indices[0]
    last_non_wake_index = non_wake_indices[-1]

    # calculate start and end of period for trimming
    start_index = max(0, first_non_wake_index - thirty_min_samples)
    end_index = min(signals.shape[1], last_non_wake_index + thirty_min_samples)

    # trim signals and labels
    trimmed_signals = signals[:, start_index:end_index]
    trimmed_labels = labels[start_index:end_index]

    return trimmed_signals, trimmed_labels


# randomly selecting x hours from each night (aligned with intervals)
def random_sample_hours(signals, labels, hours, sampling_rate=100):
    x_hour_length = int(hours * 60 * 60 * sampling_rate) # number of samples in one hour
    segment_length = 30 * sampling_rate # length of one 30s interval in samples

    total_length = len(labels) # getting number of timepoints 

    if total_length > x_hour_length:
        # calculate the number of segments that can be sampled (what is the last segment number possible so x hours fit after it)
        num_segments = (total_length - x_hour_length) // segment_length
        # choose random starting segment and multiply with segment length to get starting index (time point)
        start_idx = np.random.randint(0, num_segments+1) * segment_length
        end_idx = start_idx + x_hour_length
        signals_new = signals[:, start_idx:end_idx]
        labels_new = labels[start_idx:end_idx]

    else: 
        raise ValueError("Sampling time is longer than actual sample.")
    
    return signals_new, labels_new



# save dataset as tf.data.Dataset (5.5 hour sample) to later reimport for training, validation, and short testing
def save_dataset(data, name, directory, hours=5.5):
    def generator():
        for psg_path, hypnogram_path, age in data:
            signals, labels = load_data(psg_path, hypnogram_path)
            trimmed_signals, trimmed_labels = remove_wake(signals, labels)
            new_signals, new_labels = random_sample_hours(trimmed_signals, trimmed_labels, hours=hours)
            # yield the data as a tuple
            yield new_signals.astype(np.float32), new_labels.astype(np.int32), int(age)


    # path for saving the dataset
    path = os.path.join(directory, name)

    # create dataset from generator
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(4, None), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
    
    # Save the dataset
    tf.data.experimental.save(dataset, path)

    print(f'{name} dataset saved to {path}')


# save paths for whole night testing
def save_paths(paths, name, directory):
    # ensture directory for output file exists
    output_file = os.path.join(directory, name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # write paths to file
    path_tuples = [list(t) for t in paths] # converts tuples to lists because JSON does not support tuples
    with open(output_file, 'w') as file:
        json.dump(path_tuples, file)
    
    print(f'{name} paths saved to {output_file}')

# %%
# USE FUNCTIONS ABOVE TO PREPARE PATHS AND DATASETS
# %%
# directory which contains all of the sleep cassette data files (PSG and Hypnograms) and excel with participant data
root_path = '../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
excel_path = '../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

# load ages from excel file
age_dict = load_ages(excel_path)

# get data paths in two separate lists
psg_paths, hypnogram_paths, ages_sorted = get_data_paths(root_path, age_dict)

# split data into train, val, and test sets according to subject ID returning lists of file tuples
train, val, test = split_dataset_by_subjects(psg_paths, hypnogram_paths, ages_sorted)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# directory to save paths
paths_directory = 'split_paths_sc'

# ensure the paths directory exists
os.makedirs(paths_directory, exist_ok=True)

save_paths(train, 'train', paths_directory)
save_paths(val, 'val', paths_directory)
save_paths(test, 'test', paths_directory)

# directory to save datasets
data_directory = 'prepared_data_sc'

# ensure the dataset directory exists
os.makedirs(data_directory, exist_ok=True)

# prepare data for training, validation, and testing as tf.data.Datasets
print('PREPARING TRAIN')
save_dataset(train, 'train', data_directory)

print('PREPARING VAL')
save_dataset(val, 'val', data_directory)

print('PREPARING TEST')
save_dataset(test, 'test', data_directory)

