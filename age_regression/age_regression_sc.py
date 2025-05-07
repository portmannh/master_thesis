"""
Running age regression on SC data using different input feature types and different regression models.
This is done on separate train, validation, and test sets.
Validation set is used for selecting the best models.
"""

# %% 
# imports
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro
import datetime    
from sklearn.decomposition import PCA
import re
import tensorflow as tf
import random

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
# load data
features_dir = 'features_sc'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Load the DataFrame
# xx_data_averaged or xx_data_flattened or xx_data_correlated
use_pca = True
data_name="Flattened Features"
dt_nm = 'data_flattened'
train_df = pd.read_csv(os.path.join(features_dir, f"train_{dt_nm}.csv"))
val_df = pd.read_csv(os.path.join(features_dir, f"val_{dt_nm}.csv"))
test_df = pd.read_csv(os.path.join(features_dir, f"test_{dt_nm}.csv"))

# %%
# Separate features and ages
train_features = train_df.drop(columns=["age"]).values
train_ages = train_df["age"].values

val_features = val_df.drop(columns=["age"]).values
val_ages = val_df["age"].values

test_features = test_df.drop(columns=["age"]).values
test_ages = test_df["age"].values

# %%
if use_pca:
    dt_nm = f"{dt_nm}_pca"

#%%
# scale features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features) 
test_features_scaled = scaler.transform(test_features)

# %%
# PCA (fit on train set, transform on train, val, and test sets)
if use_pca:
    pca = PCA(n_components=10)
    train_features = pca.fit_transform(train_features_scaled)
    val_features = pca.transform(val_features_scaled)
    test_features = pca.transform(test_features_scaled)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_)}")

else:
    train_features = train_features_scaled
    val_features = val_features_scaled
    test_features = test_features_scaled

# %%
def check_normality(data):
    stat, p = shapiro(data)
    return p

def calculate_correlation(true_ages, predicted_ages):
    """Computes Pearson Correlation and Spearman Rho between true and predicted ages."""
    pearson_corr, _ = pearsonr(true_ages, predicted_ages)
    spearman_corr, _ = spearmanr(true_ages, predicted_ages)
    return pearson_corr, spearman_corr

# %%
results = []
model_predictions = {}

# %%
# Store results
def evaluate_model(name, true_ages, predicted_ages):
    normality_true = check_normality(true_ages)
    normality_pred = check_normality(predicted_ages)
    mae = mean_absolute_error(true_ages, predicted_ages)
    mse = mean_squared_error(true_ages, predicted_ages)
    brain_age_gap = predicted_ages - true_ages
    avg_brain_age_gap = np.mean(brain_age_gap)
    pearson_corr, spearman_corr = calculate_correlation(true_ages, predicted_ages)
    results.append({'model': name, 'MAE': mae, 'MSE': mse, 'Pearson': pearson_corr, 'Spearman': spearman_corr, 'Normality_true': normality_true, 'Normality_pred': normality_pred, 'bag': avg_brain_age_gap})
    model_predictions[name] = predicted_ages
    print(f'{name} MAE: {mae:.4f}, Pearson Correlation: {pearson_corr:.4f}, Normalities (true, pred): {normality_true, normality_pred}, Spearman Correlation: {spearman_corr:.4f}, Brain Age Gap: {avg_brain_age_gap:.4f}')


# Models to loop over
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'SVR (linear kernel)': SVR(kernel='linear'),
    'SVR (poly kernel)': SVR(kernel='poly'),
    'SVR (rbf kernel)': SVR(kernel='rbf'),
    'SVR (sigmoid kernel)': SVR(kernel='sigmoid')
}

# Loop over all models
for model_name, model in models.items():
    # Train model on training data
    model.fit(train_features, train_ages)
    model_predictions_for_val = model.predict(val_features)
    # Evaluate model on validation set
    evaluate_model(model_name, val_ages, model_predictions_for_val)

# %%
# Find best models based on val set
min_mae = min(result["MAE"] for result in results)
max_corr = max(result["Pearson"] for result in results)

best_mae_models = [result for result in results if result["MAE"] == min_mae]
best_corr_models = [result for result in results if result["Pearson"] == max_corr]

print("\nBest Models based on MAE:")
for model in best_mae_models:
    print(model)

print("\nBest Models based on Pearson Correlation:")
for model in best_corr_models:
    print(model)

# %%
# Add flags for the best models
best_mae_model_names = [model["model"] for model in best_mae_models]
best_corr_model_names = [model["model"] for model in best_corr_models]

# Add flags to the results
for result in results:
    result['best_for_mae'] = result['model'] in best_mae_model_names
    result['best_for_pearson'] = result['model'] in best_corr_model_names

# %%
# Save predictions of the two best test models (MAE and Pearson), avoiding duplicates
predictions_dir = 'predictions_sc'
os.makedirs(predictions_dir, exist_ok=True)

# Track which models were already saved
saved_models = set()

# Evaluate best models on test set
for best_model in best_mae_models + best_corr_models:
    model_name = best_model["model"]

    if model_name in saved_models:
        continue  # Skip if already saved

    if "Ridge" in model_name:
        model = Ridge()
    elif "SVM" in model_name:
        # Use regex to extract kernel
        match = re.search(r'SVR \((\w+) kernel\)', model_name)
        if match:
            kernel = match.group(1)
            model = SVR(kernel=kernel)
        else:
            print(f"Could not parse SVM parameters from {model_name}. Skipping...")
            continue
    else:
        model = LinearRegression()
    # Re-fit on training set and predict test set
    model.fit(train_features, train_ages)
    test_predictions = model.predict(test_features)
    evaluate_model(model_name + " (Test Set)", test_ages, test_predictions)

    df_pred = pd.DataFrame({
        'true_age': test_ages,
        'predicted_age': test_predictions
    })

    save_model_name = re.sub(r'[^\w\-_.]', '_', model_name)
    df_pred.to_csv(os.path.join(predictions_dir, f"{save_model_name}_test_predictions_{dt_nm}_{current_time}.csv"), index=False)

# %%
# Convert results to DataFrame and save results to CSV
results_df = pd.DataFrame(results)
results_dir = 'results_sc'
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(os.path.join(results_dir, f"results_{dt_nm}_{current_time}.csv"), index=False)
