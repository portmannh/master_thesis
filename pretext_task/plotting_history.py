# %% 
# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import matplotlib.font_manager
matplotlib.font_manager._rebuild()

# %%
# Name of dataset / model version
ds_name = 'SC'
# %%
# Load the training history
history_path = 'results/history_sc/20250502-111328_training_history.npy' # load correct file

history = np.load(history_path, allow_pickle=True).item()

# %%
# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# %%
colors = sns.color_palette('colorblind')
          
# Plot training & validation loss values
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation loss
axs[0].plot(history['loss'], color=colors[0], linestyle='-', linewidth=1.8, label='Training Loss')
axs[0].plot(history['val_loss'], color=colors[1], linestyle='-', linewidth=1.8, label='Validation Loss')
axs[0].set_title(f'Training and Validation Loss ({ds_name})')
axs[0].set_xlabel('Epoch (Training Iterations)')
axs[0].set_ylabel('Loss (Cross-Entropy)')
axs[0].legend(loc='lower left')
axs[0].grid(True, linestyle='--', linewidth=1.0, alpha=0.8)

# Plot training & validation accuracy
axs[1].plot((history['categorical_accuracy']), color=colors[0], linestyle='-', linewidth=1.8, label='Training Accuracy')
axs[1].plot((history['val_categorical_accuracy']), color=colors[1], linestyle='-', linewidth=1.8, label='Validation Accuracy')
axs[1].set_title(f'Training and Validation Accuracy ({ds_name})')
axs[1].set_xlabel('Epoch (Training Iterations)')
axs[1].set_ylabel('Accuracy')
axs[1].legend(loc='lower right')
axs[1].grid(True, linestyle='--', linewidth=1.0, alpha=0.8)


plt.tight_layout(pad=5.0)

plt.show()

# %%
# Save the figure
image_format = 'png'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
image_name = f'{current_time}_training_curves_sc_batch64'
directory = 'figures'
os.makedirs(directory, exist_ok=True)

fig.savefig(os.path.join(directory, f'{image_name}.{image_format}'), format=image_format, dpi=300)

