# %% 
'''
Evaluate the model (trained on the SC train set) on the entire night data contained in the test set of SC Data
'''
# %%
import tensorflow as tf
from tensorflow import keras
import json
import mne
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import os
import datetime
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
# paths to test data 
test_path = f'../data_preparation/split_paths_sc/test'

# %%
# parameters for model
num_classes = 5
batch_size = 1 # look at test data one by one 
in_channels = 4


optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['categorical_accuracy']

# %%
# load saved weights and create model
model = MyUSleep(num_classes=num_classes)

model.load_weights(f'weights/trained_sc/20250502-111328_trained_usleep_weights_normal')
model.compile(optimizer=optimizer, loss=[loss, None], metrics=[metrics, None]) # to ignore second output (features)

# %%
# load mean and variance of normalizer (adapted on train set)
mean = np.load(f'normalizer_sc/normalizer_mean.npy')
variance = np.load(f'normalizer_sc/normalizer_variance.npy')

# instantiate normalizer
normalizer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)

# %%
# FUNCTIONS TO PREPARE TEST DATA

# get list of data path tuples (as saved in json file)
def load_tuples(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return [tuple(t[:2]) for t in data] # only needing paths and not age 

# %%
# convert sleep stage (string) to numeric stage
# group stages 3 and 4 together
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


# %%
# create Dataset from generator and load data and remove wake parts
def create_dataset(data):
    def generator():
        for psg_path, hypnogram_path in data:
            signals, labels = load_data(psg_path, hypnogram_path)
            trimmed_signals, trimmed_labels = remove_wake(signals, labels)
            yield trimmed_signals.astype(np.float32), trimmed_labels.astype(np.int32)


    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(4, None), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None,), dtype=tf.int32)))
    
    return dataset

# %%
# one hot encode labels and transpose signals
def preprocessing(signals, labels):
    labels = tf.one_hot(labels, depth=num_classes)
    signals = tf.transpose(signals, perm=[1, 0])
    return signals, labels

# %%
# USE FUNCTIONS TO PREPARE TEST DATA
test = load_tuples(test_path)
test= create_dataset(test)
test = test.map(preprocessing)
test = test.batch(batch_size)
test = test.map(lambda signal, labels: (normalizer(signal), labels))

# %%
# Evaluate the model
test_results = model.evaluate(test)

# %%
test_loss, test_accuracy = test_results[0], test_results[2]
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#%% Extract predictions and labels for all test data
true_labels = []
predicted_labels = []
predicted_probabilities = []

# extract data and labels
for signal_batch, label_batch in test:
    for signal, label in zip(signal_batch, label_batch):
        signal = tf.expand_dims(signal, axis=0)
        predictions = model.predict(signal)[0] # only care about first output here
        probabilities = softmax(predictions, axis=2)
        pred_label = np.argmax(probabilities, axis=2)
        pred_label = pred_label[0]
        true_label = np.argmax(label, axis=1) # class index from one-hot encoding

        predicted_labels.append(pred_label)
        true_labels.append(true_label)
        predicted_probabilities.append(probabilities[0])

# Concatenate list of arrays
true_labels = np.concatenate(true_labels, axis=0)
predicted_labels = np.concatenate(predicted_labels, axis=0)
predicted_probabilities = np.concatenate(predicted_probabilities, axis=0)

# manually compute accuracy
man_accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
print(f'Manually computed accuracy: {man_accuracy}')

# %% compute confusion matrix for entire night

# original confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# normalized confusion matrix
# normalize by rows (sum of each row) to get ratios
row_sums = conf_matrix.sum(axis=1)
conf_matrix_normalized = np.divide(
    conf_matrix,
    row_sums[:, np.newaxis],
    out=np.zeros_like(conf_matrix, dtype=float), 
    where=row_sums[:, np.newaxis] != 0
)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# print confusion matrices (plot in different script)
print("Confusion Matrix \n:", conf_matrix)
print("Normalized Confusion Matrix \n:", conf_matrix_normalized)

# save confusion matrices
results_path = 'results/testing_sc'
os.makedirs(results_path, exist_ok=True)
np.save(os.path.join(results_path, f'{current_time}_cm_entire_night.npy'), conf_matrix)
np.save(os.path.join(results_path, f'{current_time}_cm_entire_night_normalized.npy'), conf_matrix_normalized)

# %%
# save predictions and labels for ROC and other metrics
true_labels_array = np.array(true_labels)
predicted_probabilities_array = np.array(predicted_probabilities)
predicted_labels_array = np.array(predicted_labels)

np.savez(os.path.join(results_path, f'{current_time}_results_entire_night.npz'), true_lab=true_labels_array, pred_lab= predicted_labels_array, pred_prob=predicted_probabilities_array)

print(f"Results saved to {results_path}")