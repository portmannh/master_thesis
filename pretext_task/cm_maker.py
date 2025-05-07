# %%
"""
Script to plot confusion matrix that was saved before
"""

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import datetime

# %%
# Define the path to the saved confusion matrices
conf_matrix_path = 'results/testing_upd'

ds_name = 'UPD_UPD_model'

# Load the confusion matrices
conf_matrix = np.load(os.path.join(conf_matrix_path, '20250503-105706_cm_entire_night_upd_model.npy')) 
conf_matrix_normalized = np.load(os.path.join(conf_matrix_path, '20250503-105706_cm_entire_night_normalized_upd_model.npy')) 
labels = ['Wake', 'REM', 'N1', 'N2', 'N3']

# %%
# Plot confusion matrices
# Create subplots
chosen_cmap = plt.get_cmap('cividis')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot the original confusion matrix
disp1 = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
disp1.plot(cmap=chosen_cmap, ax=axes[0], values_format='.0f', xticks_rotation=45)
axes[0].set_title(f"Confusion Matrix {ds_name}", size=18)
axes[0].set_xlabel("Predicted Label", fontsize=14)
axes[0].set_ylabel("True Label", fontsize=14)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].grid(False)

# Plot the normalized confusion matrix
disp2 = ConfusionMatrixDisplay(conf_matrix_normalized, display_labels=labels)
disp2.plot(cmap=chosen_cmap , ax=axes[1], values_format='.2f', xticks_rotation=45)
axes[1].set_title(f"Row Normalized Confusion Matrix {ds_name}", size=18)
axes[1].set_xlabel("Predicted Label", fontsize=14)
axes[1].set_ylabel("True Label", fontsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].grid(False)

# Show the plots
plt.tight_layout()
plt.subplots_adjust(wspace=0.02)

plt.show()
# %%
image_format = 'png'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
image_name = f'{current_time}_cm_{ds_name}'
directory = 'figures'
os.makedirs(directory, exist_ok=True)
fig.savefig(os.path.join(directory, f'{image_name}.{image_format}'), format=image_format, dpi=300)
# %%
