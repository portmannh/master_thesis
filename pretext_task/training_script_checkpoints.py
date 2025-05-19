# %%
""" 
Script for training the USleep model on the prepared data (SC or UPD)
Also saves mean and variance of the normalizer to be used on the test data
Saves learning curves and trained model weights 
"""

# %%
# imports
import tensorflow as tf
import numpy as np
import datetime
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_usleep_configured import MyUSleep
import glob
import re

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

# %%
# flag to determine if training set should be prepared as intended or in the same way as val set
# if True, random segment sampling and sample weights are applied
train_preparation = True 
dataset = 'sc' # dataset to be used for training (either sc or upd)

# %%
# randomly selecting T continuous samples from each night (not nececessarily 30s intervals as they were scored originally, can be shifted)
# defining the model architecture -> USleep

# %%
# Variables to set at the beginning

# setting values to determine sample size
# taken from original U-Sleep paper (Perslev et al., 2021) -> in our case this 17.5 minute interval could also be defined otherwise
T = 35 # number of 30s intervals (17.5 minutes)
sampling_rate = 100
i = 30 * sampling_rate # number of timepoints in 30 second intervals
length = int(T * i) # sample length (number of timepoints in 17.5 minutes)

# setting batch size
batch_size = 4 # changed to test once with 64 

# setting number of input and output channels
in_channels = 4 # PSG channels
num_classes = 5 # number of sleep stages (W, R, N1, N2, N3)

# variables for training
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['categorical_accuracy']
epochs = 10000 # changed to test once with 10000

# %%
# FUNCTIONS FOR PREPARATION OF TRAINING AND VALIDATION DATA

# # %%
# randomly sampling 17.5 consecutive minutes of the data (used for validation set)
def random_sample(signal, label, length):
    total_length = tf.shape(signal)[1] # getting number of timepoints 

    if total_length > length:
        start_idx = tf.random.uniform(shape=(), maxval=total_length - length, dtype=tf.int32)
        end_idx = start_idx + length
        signal = signal[:, start_idx:end_idx]
        label = label[start_idx:end_idx]
    
    return signal, label

# %%
# trying to uniformly chosse a class and sample around it at least 30s before and after instead of completely randomly choosing a sample (used for training set)
# this is done to ensure that the model sees all classes and not just the most frequent ones, however, not all classes are equally represented
def sample_segment(signal, label, length, i):

    signal = signal.numpy()
    label = label.numpy()
    total_length = signal.shape[1]
    
    if total_length > length:
        # find unique classes and their indices
        unique_classes = np.unique(label)
        class_indices = {cls: np.where(label==cls)[0] for cls in unique_classes}
        
        # ignore class -1
        if -1 in class_indices:
            del class_indices[-1]
            unique_classes = np.array([cls for cls in unique_classes if cls!=-1])

        # keep trying until valid class with valid indices is found
        while True:

            # uniformly sample a class
            sampled_class = np.random.choice(unique_classes)

            # select a random index from the sampled class
            min_start_idx = i # make sure there are 30s before idx
            max_start_idx = total_length - length # make sure there are 30s after idx
            chosen_class_indices = class_indices[sampled_class]
            valid_indices = chosen_class_indices[(chosen_class_indices <= max_start_idx) & (chosen_class_indices >= min_start_idx)]

            if len(valid_indices) > 0:
                break # exit loop if we have valid indices

        class_idx = np.random.choice(valid_indices)

        # put class index at random place in segment (not first or last 30s to ensure whole 30s segment is included)
        min_start = class_idx - (length - i)
        max_start = class_idx - i
        start_idx = tf.random.uniform(shape=(), minval=max(0, min_start) , maxval=max_start, dtype=tf.int32)
        end_idx = start_idx + length
        signal = signal[:, start_idx:end_idx]
        label = label[start_idx:end_idx]

    return signal, label


# wrapper function for sample_segment to work with tf.data.Dataset
def sample_segment_tf(signal, label, length, i):
    segment_signal, segment_label = tf.py_function(
        func=sample_segment,
        inp=[signal, label, length, i],
        Tout=(tf.float32, tf.int32)
    )
    return segment_signal, segment_label
    
# %%
# generating sample weights based on transitions
def generate_sample_weights(labels, transition_weight=0.1, default_weight=1.0, window_size=100): # one second before and after transition will be weighted less
    # create weights list
    weights = np.ones_like(labels, dtype=np.float32) * default_weight

    # find transition points
    transitions = np.where(labels[:-1] != labels[1:])[0] + 1

    # assign lower weight in window before and after transition
    for t in transitions:
        start = max(0, t - window_size)
        end = min(len(labels), t + window_size + 1)

        weights[start:end] = transition_weight
    
    undefined = np.where(labels == -1)[0] # weight 0 where sleep stage is -1 (Sleep Stage ? or Movement time) -> we don't want to classify these
    weights[undefined] = 0
    
    return weights

# apply weights to dataset
def add_weights(signal, labels):
    weights = tf.numpy_function(func=generate_sample_weights, inp=[labels], Tout=tf.float32)
    return signal, labels, weights

# %%
# apply one hot encoding to labels and transpose channels
def preprocessing(signals, labels, weights=None):
    labels = tf.one_hot(labels, depth=num_classes)
    signals = tf.transpose(signals, perm=[1, 0])
    if weights is not None:
        return signals, labels, weights
    return signals, labels

# %%
# load saved datasets
train = tf.data.experimental.load(f'../data_preparation/prepared_data_{dataset}/train')
val = tf.data.experimental.load(f'../data_preparation/prepared_data_{dataset}/val')

# %%
# get rid of age and moca (not needed for pretext task)
if dataset == 'sc':
    train = train.map(lambda signal, labels, age: (signal, labels))
    val = val.map(lambda signal, labels, age: (signal, labels))
elif dataset == 'upd':
    train = train.map(lambda signal, labels, age, moca: (signal, labels))
    val = val.map(lambda signal, labels, age, moca: (signal, labels))

# %%
# Instantiate normalizer for data normalization
normalizer = keras.layers.Normalization(axis=-1)

# get signal part of train data to adapt normalizer
def extract_signal(dataset, train_preparation):
    if train_preparation:
        return dataset.map(lambda signal, labels, weights: tf.ensure_shape(signal, [None, None, 4]))
    else:
        return dataset.map(lambda signal, labels: tf.ensure_shape(signal, [None, None, 4]))


# %%
# Prepare train data
train = train.cache()
train = train.shuffle(buffer_size=tf.data.experimental.cardinality(train).numpy(), seed=seed)

# %%
# if train_preparation is set to True, we want to sample segments around a class label
if train_preparation:
    train = train.map(lambda signal, labels: sample_segment_tf(signal, labels, length, i))
    train = train.map(lambda signal, labels: add_weights(signal, labels))
    train = train.map(lambda signal, labels, weights: preprocessing(signal, labels, weights))
else:
    train = train.map(lambda signal, labels: random_sample(signal, labels, length))
    train = train.map(lambda signal, labels: preprocessing(signal, labels))

train = train.batch(batch_size)

# %%
# adapt normalizer to train set
train_signal = extract_signal(train, train_preparation).unbatch()
normalizer.adapt(train_signal)

# %%
# save learned mean and variance
normalizer_dir = f'normalizer_{dataset}'
os.makedirs(normalizer_dir, exist_ok=True)
np.save(os.path.join(normalizer_dir, 'normalizer_mean.npy'), normalizer.mean.numpy())
np.save(os.path.join(normalizer_dir, 'normalizer_variance.npy'), normalizer.variance.numpy())

# %%
#normalize train set
if train_preparation:
    train = train.map(lambda signal, labels, weights: (normalizer(signal), labels, weights))
else:
    train = train.map(lambda signal, labels: (normalizer(signal), labels))

# %%
# Prepare validation data
val = val.map(lambda signal, labels: random_sample(signal, labels, length))
val = val.map(lambda signal, labels: preprocessing(signal, labels))
val = val.batch(batch_size)
val = val.map(lambda signal, labels: (normalizer(signal), labels))

# %%
# create model
model = MyUSleep(num_classes=num_classes)

# maybe try to pass input size to model somewhere
# Create a dummy input tensor
dummy_input = tf.random.normal(shape=(batch_size, T*i, in_channels))  # Adjust shape as needed

# Call the model with the dummy input to build it
model(dummy_input)

# compile model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.summary()

# %%
# set up TensorBoard log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# set up checkpoint callback
checkpoint_dir = f'checkpoints/{dataset}'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/{dataset}/checkpoint_{{epoch:04d}}',
    save_weights_only=False,
    save_freq='epoch',
    period=100,
    verbose=1
)

# check if there is a checkpoint to load
checkpoint_files = sorted(glob.glob(f'{checkpoint_dir}/checkpoint_*'), key=os.path.getmtime)
if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    print(f"Restoring from checkpoint: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
else:
    latest_checkpoint = None
    print("No checkpoint found. Starting training from scratch.")


# %%
# Resume training from correct epoch
initial_epoch = 0
if latest_checkpoint:
    # Extract the epoch number from the checkpoint filename
    epoch_match = re.search(r'checkpoint_(\d+)', latest_checkpoint)
    if epoch_match:
        initial_epoch = int(epoch_match.group(1))
        print(f"Resuming training from epoch: {initial_epoch}")
    else:
        print("Could not extract epoch number from checkpoint filename. Starting training from scratch.")

# %%
# defining custom callback to save history after each epoch for plotting
class ContinuousHistory(tf.keras.callbacks.Callback):
    def __init__(self, history_path):
        super().__init__()
        self.history_path = history_path
        self.history = {}
        self.start_epoch = 0

        # Load existing history if available
        if os.path.exists(self.history_path):
            self.history = np.load(self.history_path, allow_pickle=True).item()
            self.start_epoch = len(next(iter(self.history.values())))
            print(f"Loaded existing training history with {len(next(iter(self.history.values())))} epochs.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # skip if this epoch was already logged
        if epoch < self.start_epoch:
            return

        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)
        
        # save only every 100 epochs
        if epoch % 100 == 0:
            np.save(self.history_path, self.history)
            print(f"Saved training history at epoch {epoch}.")


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
history_dir = f'results/history_{dataset}'
os.makedirs(history_dir, exist_ok=True)
history_path = os.path.join(history_dir, f'training_history_10000.npy')

history_callback = ContinuousHistory(history_path)

# %%
# train model
history = model.fit(train, validation_data=val, epochs=epochs, initial_epoch=initial_epoch, callbacks=[tensorboard_callback, checkpoint_callback, history_callback])

# %%
# saving the trained model weights
weights_path = f'weights/trained_{dataset}'
os.makedirs(weights_path, exist_ok=True)

if train_preparation:
    method = 'normal'
else:
    method = 'val_preparation_method'

model.save_weights(os.path.join(weights_path, f'{current_time}_trained_usleep_weights_{method}'))
