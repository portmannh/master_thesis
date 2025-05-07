# %%
"""
Create ROC curves for multiclass classification and compute AUC
1 vs 1 and 1 vs Rest
"""

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import seaborn as sns

# %%
# load existing data, true_lab (int) and pred_prob (5 probabilities)
directory = 'results/testing_upd'
data = np.load(os.path.join(directory, '20250503-105706_results_entire_night_upd_model.npz')) 
y_true = data['true_lab']
y_probs = data['pred_prob']

dataset_name = 'UPD'
# %%
"""
One vs Rest ROC for all classes
"""

# %%
# Get the number of classes
class_names = ["Wake", "REM", "N1", "N2", "N3"]

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette('colorblind')

# Iterate through each class for One-vs-Rest
for i, class_name in enumerate(class_names):
    # Binarize labels for the current class
    y_true_binary = (y_true == i).astype(int)  # 1 for the current class, 0 for the rest
    
    # Use predicted probabilities for the current class
    y_probs_binary = y_probs[:, i]
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_probs_binary)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_name} vs. Rest (AUC = {roc_auc:.2f})')

# Plot the baseline
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guessing')

ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title(f'One-vs-Rest ROC Curves {dataset_name}', fontsize=16)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

plt.show()

# %%
image_format = 'png'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
image_name = f'{current_time}_OVR_ROC_curves_{dataset_name}'
directory = 'figures'
os.makedirs(directory, exist_ok=True)
fig.savefig(os.path.join(directory, f'{image_name}.{image_format}'), format=image_format, dpi=300)

# %%
"""
One vs One AUC for all classes
"""

# %%
# Compute and print AUC for all OvO comparisons
classes = np.unique(y_true)
n_classes = len(classes)

print("One-vs-One AUC scores:")
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        class_a = classes[i]
        class_b = classes[j]
        
        # Filter to include only Class A and Class B
        binary_mask = (y_true == class_a) | (y_true == class_b)
        y_true_binary = (y_true[binary_mask] == class_a).astype(int)  # 1 for Class A, 0 for Class B
        
        # Normalize probabilities for Class A and Class B
        y_probs_binary = y_probs[binary_mask][:, [class_a, class_b]]  
        y_probs_normalized = y_probs_binary / y_probs_binary.sum(axis=1, keepdims=True)
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs_normalized[:, 0])
        roc_auc = auc(fpr, tpr)
        
        # Print the AUC score
        print(f'Class {class_a} vs. Class {class_b}: AUC = {roc_auc:.2f}')

        # do the same the other way around
        # Filter to include only Class A and Class B
        binary_mask = (y_true == class_a) | (y_true == class_b)
        y_true_binary = (y_true[binary_mask] == class_b).astype(int)  # 1 for Class A, 0 for Class B

        # Normalize probabilities for Class A and Class B
        y_probs_binary = y_probs[binary_mask][:, [class_b, class_a]] 
        y_probs_normalized = y_probs_binary / y_probs_binary.sum(axis=1, keepdims=True)

        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs_normalized[:, 0])
        roc_auc = auc(fpr, tpr)
        
        # Print the AUC score
        print(f'Class {class_b} vs. Class {class_a}: AUC = {roc_auc:.2f}')
# %%
