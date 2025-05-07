# %%
'''
Evaluate the model on the entire night data contained in the upd dataset / or just test part if evaluating newly trained model 
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
import pandas as pd
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
# determine if using sc model with all upd data or upd model with only test data
transfer = False # True if using sc model, False if using upd model

# # %%
# paths to test data
if transfer:
    test_path = '../data_preparation/split_paths_upd/healthy'
else:
    test_path = '../data_preparation/split_paths_upd/test'

# %%
# parameters for model
num_classes = 5
batch_size = 1 # look at test data one by one 
in_channels = 4

optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['categorical_accuracy']

# %%
# load save weights and create model
model = MyUSleep(num_classes=num_classes)

if transfer:
    weights_path = 'weights/trained_sc/20250502-111328_trained_usleep_weights_normal'
else:
    weights_path = 'weights/trained_upd/20250428-142901_trained_usleep_weights_normal'
model.load_weights(weights_path)

model.compile(optimizer=optimizer, loss=[loss, None], metrics=[metrics, None]) # to ignore second output (features)

# %%
# get signal part of train data to adapt normalizer
def extract_signal(dataset):
    return dataset.map(lambda signal, labels: tf.ensure_shape(signal, [None, None, 4]))

# %%
# FUNCTIONS TO PREPARE TEST DATA
# get list of data path tuples (as saved in json file)
def load_tuples(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return [t[0] for t in data]

# convert sleep stage (string) to numeric stage
def map_sleep_stage_to_label(stage):
    stage_mapping = {
        'W' : 0,
        'R' : 1,
        'N1' : 2,
        'N2' : 3,
        'N3' : 4
    }
    return stage_mapping.get(stage, -1) # will be -1 if none of the above

# %%
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
# create Dataset from generator
# only containing signals and labels as age and moca are not needed here
def create_dataset(data):
    def generator():
        for path in data:
            signals, labels = load_data(path)
            trimmed_signals, trimmed_labels = remove_wake(signals, labels)
            yield trimmed_signals.astype(np.float32), trimmed_labels.astype(np.int32)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(4, None), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None,), dtype=tf.int32)))
    
    return dataset


# one-hot encode labels and transpose signals
def preprocessing(signals, labels):
    labels = tf.one_hot(labels, depth=num_classes)
    signals = tf.transpose(signals, perm=[1, 0])
    return signals, labels

# %%
data = load_tuples(test_path)
data = create_dataset(data)
data = data.map(preprocessing)
data = data.batch(batch_size)

# %%
# adapt normalizer to data set
if transfer:
    normalizer = keras.layers.Normalization(axis=-1)
    data_signal = extract_signal(data).unbatch()
    normalizer.adapt(data_signal)
    data = data.map(lambda signal, labels: (normalizer(signal), labels))
else:
    mean = np.load('normalizer_upd/normalizer_mean.npy')
    variance = np.load('normalizer_upd/normalizer_variance.npy')
    normalizer = keras.layers.Normalization(axis=-1, mean=mean, variance=variance)
    data = data.map(lambda signal, labels: (normalizer(signal), labels))

# %%
# Evaluate the model
test_results = model.evaluate(data)
test_loss, test_accuracy = test_results[0], test_results[2]
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#%% manually computed accuracy overall
true_labels = []
predicted_labels = []
predicted_probabilities = []

# extract data and labels
for signal_batch, label_batch in data:
    for signal, label in zip(signal_batch, label_batch):
        signal = tf.expand_dims(signal, axis=0)
        predictions = model.predict(signal)[0] # only care about first output here
        probabilities = softmax(predictions, axis=2)
        pred_label = np.argmax(probabilities, axis=2)
        pred_label = pred_label[0]
        true_label = np.argmax(label, axis=1)

        predicted_labels.append(pred_label)
        true_labels.append(true_label)
        predicted_probabilities.append(probabilities[0])

# concatenate lists of arrays
true_labels = np.concatenate(true_labels, axis=0)
predicted_labels = np.concatenate(predicted_labels, axis=0)
predicted_probabilities = np.concatenate(predicted_probabilities, axis=0)

# manually compute accuracy
man_accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
print(f'Manually computed accuracy: {man_accuracy}')

# %%
# compute overall confusion matrix

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

if transfer:
    model_name = 'sc_model'
else:
    model_name = 'upd_model'

# save confusion matrices
results_path = 'results/testing_upd'
os.makedirs(results_path, exist_ok=True)
np.save(os.path.join(results_path, f'{current_time}_cm_entire_night_{model_name}.npy'), conf_matrix)
np.save(os.path.join(results_path, f'{current_time}_cm_entire_night_normalized_{model_name}.npy'), conf_matrix_normalized)

# %%
# save predictions and labels for ROC and other metrics
true_labels_array = np.array(true_labels)
predicted_probabilities_array = np.array(predicted_probabilities)
predicted_labels_array = np.array(predicted_labels)

np.savez(os.path.join(results_path, f'{current_time}_results_entire_night_{model_name}.npz'), true_lab=true_labels_array, pred_lab= predicted_labels_array, pred_prob=predicted_probabilities_array)