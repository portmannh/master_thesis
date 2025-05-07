"""
Plotting the results of the age regression models
Scatterplot of true vs. predicted ages
Residuals plot of prediction errors
"""
# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import seaborn as sns

# %%
# Load saved predictions and set folder for figures
predictions_dir = 'predictions_upd'
chosen_file = 'SVR_(linear_kernel)_averaged_pca_20250506-110026.csv'
filepath = os.path.join(predictions_dir, chosen_file)

figures_dir = 'figures_upd'
os.makedirs(figures_dir, exist_ok=True)
image_format = 'png'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Specify model and data names
base_model_name = 'SVR (linear kernel)'
data_name = 'PCA Averaged Features'

sns.set(context='paper', style='whitegrid', palette='colorblind')

plt.rcParams.update({
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

palette = sns.color_palette("colorblind")

# Functions for plots and correlations
def plot_predictions(true_ages, predicted_ages, method_name="Regression Model", data_name=None):
    """Plots True vs. Predicted Age Scatterplot."""
    fig, ax = plt.subplots(figsize=(6,6))   
    ax.scatter(true_ages, predicted_ages, alpha=0.5, label="Predictions", color=palette[0])
    ax.plot([min(true_ages), max(true_ages)], [min(true_ages), max(true_ages)], 'r--', label="Ideal Fit (y=x)", color=palette[1])
    ax.set_xlabel("True Age (years)")
    ax.set_ylabel("Predicted Age (years)")
    ax.set_title(f"True vs. Predicted Age ({method_name}) \n with {data_name}")
    ax.legend()
    ax.grid(True)
    plt.show()
    return fig

def plot_residuals(true_ages, predicted_ages, method_name="Regression Model", data_name=None):
    """Plots Residuals (Errors) vs. True Age."""
    errors = predicted_ages - true_ages
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(true_ages, errors, alpha=0.5, color=palette[0])
    ax.axhline(0, color=palette[1], linestyle='--')
    ax.set_xlabel("True Age (years)")
    ax.set_ylabel("Prediction Error (years)")
    ax.set_title(f"Residual Plot ({method_name}) \n with {data_name}")
    ax.grid(True)
    plt.show()
    return fig

# %%
# Load the model predictions from the CSV file
df = pd.read_csv(filepath)

# %%
# Plot test set results for the best models

method_name = base_model_name + " (LOO CV)"

fig1 = plot_predictions(df['true_age'], df['predicted_age'],
                        method_name=method_name, data_name=data_name)
fig1.savefig(os.path.join(figures_dir, f'predictions_vs_true_{base_model_name.replace(" ", "_")}_{current_time}.{image_format}'),
             format=image_format, dpi=300)

fig2 = plot_residuals(df['true_age'], df['predicted_age'],
                      method_name=method_name, data_name=data_name)
fig2.savefig(os.path.join(figures_dir, f'residuals_{base_model_name.replace(" ", "_")}_{current_time}.{image_format}'),
             format=image_format, dpi=300)
# %%
