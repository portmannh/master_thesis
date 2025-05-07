# %%
"""
Preparing UPD data as:
- paths for whole night testing (pretext task)
- 3 hour tf.data.Dataset for training, validation, and 17.5 min pre-testing (pretext task)
- including age and MOCA score for downstream tasks

Train/val/test split for pretext task: grouped by subject ID (only healthy subjects)
MCI/Healthy split for downstream task: based on MOCA score (not split again because LOO-CV is used)

2h 17m
""" 

# %%
# imports
import mne
import numpy as np
import os
import tensorflow as tf
import re
from collections import defaultdict
import pandas as pd
import json
import random

# set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# %%
# FUNCTIONS TO RETRIEVE AND PREPARE DATA

# %%
# Finding the excel sheet in the data directory
def get_excel_file(directory):
    for filename in os.listdir(directory):
        if filename == 'vars_of_interest.xlsx':
            file_path = os.path.join(directory, filename)
            return file_path
    raise FileNotFoundError("Correct Excel file not found in directory")

# %%
# loading the age and MOCA score of participants from the excel sheet
def load_excel_data(path):
    excel_path = get_excel_file(path)
    descriptions = pd.read_excel(excel_path, engine='openpyxl')
    age_dict = {row['ID']: row['age'] for _, row in descriptions.iterrows()}
    moca_dict = {row['ID']: row['MOCA_score'] for _, row in descriptions.iterrows()}
    return age_dict, moca_dict

# %%
# get subject ID including study number
def extract_subject_code(path):
    pattern = r'\d{2}-\d{2}-\d{3}'
    match = re.search(pattern, path)
    if match:
        return match.group(0)
    return None

# %%
# get a list of all .set files
def get_data_paths(directory, age_dict, moca_dict):

    paths = []
    ages = []
    mocas = []

    # go through all sub directorie (19, 21 etc.)
    for d in os.listdir(directory):

        d_path = os.path.join(directory, d)

        if os.path.isdir(d_path):
            all_files = os.listdir(d_path)

            files = [f for f in all_files if f.endswith('.set')]

            for file in files:
                subject_code = extract_subject_code(file)
                paths.append(os.path.join(d_path, file))
                ages.append(age_dict.get(subject_code, -1))
                mocas.append(moca_dict.get(subject_code, -1))

    return paths, ages, mocas

# %% 
# Function to extract subject ID from file path (only three digit ID)
def extract_subject_id(path):
    # extract subject number from file name
    pattern = r'\d{2}-\d{2}-(\d{3})-'
    match = re.search(pattern, path)
    if match:
        return int(match.group(1))
    return None

# %%
# Function to flatten the list of subject files
# ignores the subject number that was used to sort the files
def flatten_subject_files(subject_list):
    return [file_tuple for subject, files in subject_list for file_tuple in files]

# %%
# Function to split the data into healthy and MCI subjects based on MOCA score
def split_moca(paths, ages, mocas):
    healthy_subject_files = []
    mci_subject_files = []

    for path, age, moca in zip(paths, ages, mocas):
        if moca >= 26:
            healthy_subject_files.append((path, age, moca))
        else:
            mci_subject_files.append((path, age, moca))
    
    return healthy_subject_files, mci_subject_files

# %%
# Function to split data based on subject ID
def split_dataset_by_subject(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # create a dictionary to group files by subject ID (this creates per default an empty list per key)
    subject_files = defaultdict(list)

    for path, age, moca in data:
        subject_id = extract_subject_id(path)

        # append file tuple to corresponding subject
        subject_files[subject_id].append((path, age, moca))
    
    # convert dictionary to list and shuffle
    subject_list = list(subject_files.items())
    np.random.shuffle(subject_list)

    # calculate number of subjects for each split
    num_subjects = len(subject_list)
    train_end = int(train_ratio * num_subjects)
    val_end = train_end + int(val_ratio * num_subjects)

    # split subjects into train, val, and test
    train_subjects = subject_list[:train_end]
    val_subjects = subject_list[train_end:val_end]
    test_subjects = subject_list[val_end:]

    # convert list of subjects and file tuples to list of file tuples (now not in lists ordered by subject anymore)
    train_files = flatten_subject_files(train_subjects)
    val_files = flatten_subject_files(val_subjects)
    test_files = flatten_subject_files(test_subjects)

    return train_files, val_files, test_files


# %%
# convert sleep stage (string) to numeric stage
def map_sleep_stage_to_label(stage):
    stage_mapping = {
        'W' : 0,
        'R' : 1,
        'N1' : 2,
        'N2' : 3,
        'N3' : 4
    }
    return stage_mapping.get(stage, -1) # will be -1 if none of the above (have to check if there are any)

# %%
# load data from .set file
def load_data(path):

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='warning')

    # choose channels I want to use (equivalent to those from Sleep-EDF dataset)
    channels = ['E15', 'E62', 'E75', 'E126', 'E127', 'E120']
    raw.pick_channels(channels)

    # down-sample to 100 Hz
    target_sfreq = 100
    raw.resample(target_sfreq, npad="auto", verbose='warning')

    eeg_data = raw.get_data()

    # combine channels to get equivalent to Sleep-EDF dataset
    combined_E62_E75 = eeg_data[1] - eeg_data[2] # corresponds to Pz-Oz
    combined_E126_E127 = eeg_data[3] - eeg_data[4] # corresponds to horizontal EOG

    # select final 4 channels
    eeg_data = np.vstack([eeg_data[0], combined_E62_E75, combined_E126_E127, eeg_data[5]])

    annotations_df = pd.DataFrame({
        'onset': raw.annotations.onset,
        'duration': raw.annotations.duration,
        'description': raw.annotations.description
    })

    # Sleep stages to keep
    sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
    # Filter dataframe
    filtered_df = annotations_df[annotations_df['description'].isin(sleep_stages)]

    # Calculate difference between consecutive onsets
    filtered_df = filtered_df.copy()
    filtered_df['duration'] = filtered_df['onset'].diff().shift(-1)
    filtered_df['duration'].fillna(0, inplace=True)

    n_samples = len(raw.times)
    sample_rate = raw.info['sfreq']

    sleep_stage_labels = np.zeros(n_samples, dtype=np.int32)

    # create list containing sleep stage for every time point
    for _, row in filtered_df.iterrows():
        onset_sample = int(row['onset'] * sample_rate)
        duration_sample = int(row['duration'] * sample_rate)
        stage = row['description']

        stage_label = map_sleep_stage_to_label(stage)

        sleep_stage_labels[onset_sample:onset_sample + duration_sample] = stage_label

    return(eeg_data, sleep_stage_labels)

# %%
# removing wake stages more than 30 minutes before and after sleep
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

# %%
# randomly selecting a number of hours from each night
def random_sample_hours(signals, labels, hours, sampling_rate=100):
    x_hour_length = int(hours * 60 * 60 * sampling_rate)

    total_length = len(labels)

    if total_length > x_hour_length:
        start_idx = np.random.randint(0, total_length - x_hour_length)
        end_idx = start_idx + x_hour_length
        signals_new = signals[:, start_idx:end_idx]
        labels_new = labels[start_idx:end_idx]

    else:
        raise ValueError("Sampling time is longer than actual sample.")
    
    return signals_new, labels_new

# %%
# save dataset as tf.data.Dataset (5.5 hour sample) to later reimport for training, validation, and short testing
def save_dataset(data, name, directory, hours=3):
    def generator():
        for path, age, moca in data:
            signals, labels = load_data(path)
            trimmed_signals, trimmed_labels = remove_wake(signals, labels)
            new_signals, new_labels = random_sample_hours(trimmed_signals, trimmed_labels, hours=hours)
            yield new_signals.astype(np.float32), new_labels.astype(np.int32), int(age), int(moca)

    # path for saving the dataset
    path = os.path.join(directory, name)

    # create dataset from generator
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(4, None), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(), dtype=tf.int32),
                                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
    # save the dataset
    tf.data.experimental.save(dataset, path)

    print(f'{name} dataset saved to {path}')

# %%
# save paths for whole night testing
def save_paths(paths, name, directory):
    output_file = os.path.join(directory, name)

    # write paths to file
    path_tuples = [list(t) for t in paths] # converts tuples to lists because JSON does not support tuples
    with open(output_file, 'w') as file:
        json.dump(path_tuples, file)
    
    print(f'{name} paths saved to {output_file}')

# %%
# USE FUNCTIONS ABOVE TO PREPARE PATHS AND DATASETS
# %%
# directory containing upd data
root_path = '../../data/upd_data'

# %%
# load age and MOCA score from excel sheet
age_dict, moca_dict = load_excel_data(root_path)

# %%
# get data paths and corresponding age and MOCA score
paths, ages, mocas = get_data_paths(root_path, age_dict, moca_dict)

# %%
# split data into healthy and MCI subjects based on MOCA score
healthy, mci = split_moca(paths, ages, mocas)
random.shuffle(healthy)
random.shuffle(mci)
print(f'Healthy subjects: {len(healthy)}, MCI subjects: {len(mci)}')

# %%
# split healthy subjects into train, val, and test sets
train, val, test = split_dataset_by_subject(healthy)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

# %%
paths_directory = 'split_paths_upd'
os.makedirs(paths_directory, exist_ok=True)
# %%
save_paths(train, 'train', paths_directory)
save_paths(val, 'val', paths_directory)
save_paths(test, 'test', paths_directory)
save_paths(mci, 'mci', paths_directory)
save_paths(healthy, 'healthy', paths_directory)

# %%
# directory to save the datasets
directory = 'prepared_data_upd'

# ensure the directory exists
os.makedirs(directory, exist_ok=True)

print('PREPARING TRAIN')
save_dataset(train, 'train', directory)

print('PREPARING VAL')
save_dataset(val, 'val', directory)

print('PREPARING TEST')
save_dataset(test, 'test', directory)

print('PREPARING MCI')
save_dataset(mci, 'mci', directory)

print('PREPARING HEALTHY')
save_dataset(healthy, 'healthy', directory)

