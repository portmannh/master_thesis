# %%
"""
Script to extract and save learned features from SC data from the bottleneck layer of the USleep model
Features are extracted and saved as the original train, val and test sets
"""

# %%
# imports
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_usleep import MyUSleep


# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
# paths to all prepared sc data (5.5 hour segments)
train_path = '../data_preparation/prepared_data_sc/train'
val_path   = '../data_preparation/prepared_data_sc/val'
test_path = '../data_preparation/prepared_data_sc/test'

# %%
# load mean and variance of normalizer (adapted on train set during pretext task training)
mean = np.load('../pretext_task/normalizer_sc/normalizer_mean.npy')
variance = np.load('../pretext_task/normalizer_sc/normalizer_variance.npy')

# instantiate normalizer
normalizer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)

# parameters for model
num_classes = 5
batch_size = 1 # look at test data one by one
in_channels = 4 # number of channels

# %%
# Load saved datasets
train = tf.data.experimental.load(train_path)
val = tf.data.experimental.load(val_path)
test = tf.data.experimental.load(test_path)

# %%
# one hot encode labels and transpose signals
def preprocessing(signals, labels, age):
    labels = tf.one_hot(labels, depth=num_classes)
    signals = tf.transpose(signals, perm=[1, 0])
    return signals, labels, age

# %%
# USE FUNCTIONS TO PREPARE DATA
train = train.map(preprocessing)
train = train.batch(batch_size)
train = train.map(lambda signal, labels, age: (normalizer(signal), labels, age))

val = val.map(preprocessing)
val = val.batch(batch_size)
val = val.map(lambda signal, labels, age: (normalizer(signal), labels, age))

test = test.map(preprocessing)
test = test.batch(batch_size)
test = test.map(lambda signal, labels, age: (normalizer(signal), labels, age))

# %%
# load saved weights and create model
model = MyUSleep(num_classes=num_classes, in_shape=(None, in_channels))
model.load_weights('../pretext_task/weights/trained_sc/20250502-111328_trained_usleep_weights_normal')

timepoints_n = int(100 * 60 * 5.5 * 60) # 5.5 hours of data
# Create a dummy input tensor
dummy_input = tf.random.normal(shape=(batch_size, timepoints_n, in_channels))  
# Call the model with the dummy input to build it
model(dummy_input)

model.summary()

# %%
# extract features
# Initialize dictionaries to store features and ages for each dataset
features = {}
ages = {}

# Extract features and ages for each dataset
datasets = {"train": train, "val": val, "test": test}

for dataset_name, dataset in datasets.items():
    features_list = []
    ages_list = []

    for signals, labels, age in dataset:
        # Extract bottleneck features
        _,  bottleneck_features = model(signals, training=False)

        features_list.append(bottleneck_features)
        ages_np = age.numpy().reshape(-1) # ensure age is 1D array
        ages_list.extend(ages_np)

    # Convert lists to arrays and store them in dictionaries
    features[f"{dataset_name}"] = np.vstack(features_list)
    ages[f"{dataset_name}"] = np.array(ages_list)

    # Check the shapes of the extracted features and ages
    print(f"{dataset_name} features shape: {features[dataset_name].shape}")
    print(f"{dataset_name} ages shape: {ages[dataset_name].shape}")

# %%
# option 1: flatten features (per subject)
flattened_features = {}
for dataset_name, dataset_features in features.items():
    flattened_features[dataset_name] = dataset_features.reshape(dataset_features.shape[0], -1) 
    print(f"{dataset_name} flattened features shape: {flattened_features[dataset_name].shape}")

# %%
# option 2: average each feature (per subject)
averaged_features = {}
for dataset_name, dataset_features in features.items():
    averaged_features[dataset_name] = np.mean(dataset_features, axis=1)
    print(f"{dataset_name} averaged features shape: {averaged_features[dataset_name].shape}")


# %%
# save averaged features and ages
features_dir = 'features_sc'
os.makedirs(features_dir, exist_ok=True)

for dataset_name in averaged_features.keys():
    # combine features and ages into a DataFrame
    dataset_features = averaged_features[f"{dataset_name}"]
    dataset_ages = ages[f"{dataset_name}"]

    # create DataFrame
    df1 = pd.DataFrame(dataset_features)
    df1['age'] = dataset_ages

    # save DataFrame to CSV file
    df1.to_csv(os.path.join(features_dir, f"{dataset_name}_data_averaged.csv"), index=False)

print(f'Averaged Features and ages saved to {features_dir}')

# %%
# save flattened features and ages
for dataset_name in flattened_features.keys():
    # combine features and ages into a DataFrame
    dataset_features = flattened_features[f"{dataset_name}"]
    dataset_ages = ages[f"{dataset_name}"]

    # create DataFrame
    df2 = pd.DataFrame(dataset_features)
    df2['age'] = dataset_ages

    # save DataFrame to CSV file
    df2.to_csv(os.path.join(features_dir, f"{dataset_name}_data_flattened.csv"), index=False)

print(f'Flattened Features and ages saved to {features_dir}')

