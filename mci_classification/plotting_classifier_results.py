
"""
# This script is used to plot the results of the classification models.
# It generates ROC curves and confusion matrices for the selected models
"""

# %%
# Imports
import ast 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# %%
# Set seaborn style
sns.set(style='whitegrid' ,palette='colorblind')

# Load results
dt_nm = 'averaged'  # options: 'averaged', 'flattened', 'averaged_pca', 'flattened_pca'
results_dir = 'results'
best_model = pd.read_csv(os.path.join(results_dir, 'best_auc_models_averaged_20250507-145454.csv'))

# %%
# Convert stringified lists back to real lists
list_cols = ["FPR", "TPR", "y_true", "y_pred"]

for col in list_cols:
    best_model[col] = best_model[col].apply(ast.literal_eval)
   
# %%
# Set up plotting
image_format = 'png'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
directory = 'figures'
os.makedirs(directory, exist_ok=True)

labels = ['Healthy', 'MCI']

# Define a function for plotting ROC curves
def plot_roc(fpr, tpr, auc_score, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name} ({dt_nm} Features)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"roc_{model_name}_{dt_nm}_{current_time}.{image_format}"), format=image_format, dpi=300)
    plt.show()

# Define a function for plotting confusion matrices
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.grid(False)
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=True)  # Add minor ticks for cell borders
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=True)
    ax.grid(which="minor", color="lightgrey", linestyle='-', linewidth=1)  # Add cell borders
    ax.tick_params(which="minor", size=0)  # Hide minor ticks
    plt.title(f'Confusion Matrix - {model_name} ({dt_nm} Features)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"cm_{model_name}_{dt_nm}_{current_time}.{image_format}"), format=image_format, dpi=300)
    plt.show()

# Plot and save ROC & Confusion Matrix for best model
for _, row in best_model.iterrows():
    model_name = row['Model']
    plot_roc(row['FPR'], row['TPR'], row['AUC'], model_name)
    plot_confusion_matrix(row['y_true'], row['y_pred'], model_name)
 
# %%
