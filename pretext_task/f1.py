# %%
"""
Computing F1 Score for Multi-Class Classification from existing true and predicted labels
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

# %%
# loading saved data
directory = 'results/testing_upd'
data = np.load(os.path.join(directory, '20250503-105706_results_entire_night_upd_model.npz')) # results_entire_night.npz # results_segments.npz
y_true = data['true_lab']
y_pred = data['pred_lab']

# %%
# Compute F1 Score for each class separately
def f1_metric_per_class(y_true, y_pred, num_classes):
    """
    Computes the F1 score for each class in a multi-class classification setting.
    """

    # Convert y_true and y_pred to float32 for calculation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Initialize variables to hold the per-class F1 scores
    f1_scores = []

    # Iterate over each class
    for i in range(num_classes):
        # Create binary masks for the current class
        y_true_class = tf.cast(tf.equal(y_true, i), tf.float32) # 1 if true label is i, 0 otherwise
        y_pred_class = tf.cast(tf.equal(y_pred, i), tf.float32) # 1 if predicted label is i, 0 otherwise

        # Calculate true positives, false positives, and false negatives
        true_positives = tf.reduce_sum(y_true_class * y_pred_class)
        false_positives = tf.reduce_sum((1 - y_true_class) * y_pred_class)
        false_negatives = tf.reduce_sum(y_true_class * (1 - y_pred_class))

        # Calculate precision and recall for the current class
        precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon()) # epsilon to avoid division by zero (1e-07)
        recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

        # Calculate F1 score for the current class
        f1_class = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        f1_scores.append(f1_class)

    # Return the F1 score for each class
    return tf.stack(f1_scores)

# %%
def overall_f1_score(y_true, y_pred, num_classes, average='macro'):
    """
    Computes the overall F1 score, either macro or weighted, for multi-class classification.
    """
    # Get individual class F1 scores
    f1_per_class = f1_metric_per_class(y_true, y_pred, num_classes)

    if average == 'macro':
        # Macro F1 score: average of individual class F1 scores
        return tf.reduce_mean(f1_per_class)

    elif average == 'weighted':
        # Weighted F1 score: average F1 score weighted by support (number of true samples)
        # Calculate number of true samples for each class
        support = []
        for i in range(num_classes):
            support_i = tf.reduce_sum(tf.cast(tf.equal(y_true, i), tf.float32))
            support.append(support_i)
        
        # Calculate weighted F1 score
        support = tf.stack(support)
        weighted_f1 = tf.reduce_sum(support * f1_per_class) / tf.reduce_sum(support)
        return weighted_f1

    else:
        raise ValueError("Unknown average method. Use 'macro' or 'weighted'.")

# %%
f1_classes = f1_metric_per_class(y_true, y_pred, num_classes=5)
f1_overall = overall_f1_score(y_true, y_pred, num_classes=5, average='weighted')

print(f'F1 per class: {f1_classes}')
print(f'F1 weighted overall: {f1_overall}')
# %%
