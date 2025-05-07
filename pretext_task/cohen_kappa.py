# %%
"""
Script to compute Cohen's Kappa over 30 second intervals for multi-class classification.
Also computes the interval accuracy and confusion matrix over intervals.
Only use this with SC data as UPD data is not in 30s intervals after preprocessing (due to cut out segments in processing)
"""

# %%
import numpy as np
import tensorflow as tf
import os

# %%
# loading true labels and predictions
directory = 'results/testing_sc'
data = np.load(os.path.join(directory, '20250502-142755_results_entire_night.npz')) # results_entire_night.npz # results_segments.npz
y_true = data['true_lab']
y_pred = data['pred_lab']

# %%
# aggregate labels to intervals (30s)
def aggregate_to_intervals(labels, interval_length=30, sampling_rate=100):
    points_per_interval = interval_length * sampling_rate
    intervals = len(labels) // points_per_interval
    aggregated_labels = [
        np.bincount(labels[i * points_per_interval:(i + 1) * points_per_interval]).argmax()
        for i in range(intervals)
    ]
    return np.array(aggregated_labels)

# %%
y_true_intervals = aggregate_to_intervals(y_true)
y_pred_intervals = aggregate_to_intervals(y_pred)

# %%
def cohen_kappa(y_true, y_pred, num_classes):
    """
    Computes Cohen's Kappa for multi-class classification.
    
    Parameters:
    - y_true: Ground truth labels (1D array of integers).
    - y_pred: Predicted labels (1D array of integers).
    - num_classes: Number of unique classes.
    
    Returns:
    - kappa: Cohen's Kappa score.
    """
    # Compute the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    print(confusion_matrix) # show confusion matrix over intervals
    confusion_matrix = tf.cast(confusion_matrix, tf.float32)

    # Total number of samples
    total_samples = tf.reduce_sum(confusion_matrix)

    # Observed agreement (Po): diagonal elements / total samples
    observed_agreement = tf.reduce_sum(tf.linalg.diag_part(confusion_matrix)) / total_samples

    # Expected agreement (Pe): compute row and column marginals
    row_marginals = tf.reduce_sum(confusion_matrix, axis=1)  # Sum across rows
    col_marginals = tf.reduce_sum(confusion_matrix, axis=0)  # Sum across columns
    expected_agreement = tf.reduce_sum(row_marginals * col_marginals) / (total_samples ** 2)

    # Compute Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement + tf.keras.backend.epsilon()) # epsilon to avoid division by zero (1e-07)

    return kappa

# %%
kappa = cohen_kappa(y_true_intervals, y_pred_intervals, num_classes=5)
print("Cohen's Kappa:", kappa.numpy())

# %%
acc_intervals = np.mean(y_true_intervals == y_pred_intervals)
print("Interval accuracy:", acc_intervals)
# %%
