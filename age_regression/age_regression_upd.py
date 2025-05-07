"""
Age regression on upd data using LOO CV as the dataset is very small
Different input feature types are used: averaged, flattened, PCA, and correlated features
Only data from healthy subjects is used
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
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
import random
import tensorflow as tf

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
# Select type of features to use
features_set = 'correlated_averaged' # options: 'averaged', 'flattened', 'averaged_pca', 'flattened_pca', 'correlated_averaged'
pca_used = 'pca' in features_set
correlations_used = 'correlated' in features_set

data_load_averaged = 'averaged' in features_set
if data_load_averaged:
    data_name = 'averaged'
else:
    data_name = 'flattened'

# %%
# Load data
features_path = f'features_upd/healthy_data_{data_name}.csv'

# Load the DataFrame
healthy_df = pd.read_csv(features_path)
shuffled_df = healthy_df.sample(frac=1, random_state=seed)

if pca_used:
    data_name = f"{data_name}_pca"
elif correlations_used:
    data_name = f"{data_name}_correlated"

# make sure not to use PCA and correlation-based selection at the same time
assert not (pca_used and correlations_used), "Don't use PCA and correlation-based selection at the same time."

# %%
# Separate features and ages
features = shuffled_df.drop(columns=["age", "moca"]).values # moca score not used
ages = shuffled_df["age"].values

# %% 
# Functions for correlations and evaluations
def calculate_correlation(true_ages, predicted_ages):
    """Computes Pearson r and Spearman rho Correlation between true and predicted ages."""
    pearson_corr, _ = pearsonr(true_ages, predicted_ages)
    spearman_corr, _ = spearmanr(true_ages, predicted_ages)
    return pearson_corr, spearman_corr

def check_normality(data):
    stat, p = shapiro(data)
    return p

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

# %% 
# Leave-One-Out Cross-Validation (LOO-CV)
loo = LeaveOneOut()
results = []
model_predictions = {}

# Models to loop over
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'SVR (linear kernel)': SVR(kernel='linear'),
    'SVR (poly kernel)': SVR(kernel='poly'),
    'SVR (rbf kernel)': SVR(kernel='rbf'),
    'SVR (sigmoid kernel)': SVR(kernel='sigmoid')
}

# %% 
# Loop over all models
for model_name, model in models.items():
    model_predictions_for_fold = np.zeros_like(ages, dtype=float)
    
    for train_index, test_index in loo.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = ages[train_index], ages[test_index]
        
        # Scale data using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA if needed
        if pca_used:
            pca = PCA(n_components=10)  # Select the number of components
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        # Feature selection based on the top 10 features most correlated with age
        if correlations_used:
            # Check normality of age
            stat, p_age = shapiro(y_train)
            print(f"Shapiro-Wilk test for 'age': p-value = {p_age:.4f} -> {'Normal' if p_age > 0.05 else 'Not normal'}")
            # Check normality of features
            feature_cols = X_train_scaled.shape[1]
            non_normal_features = []
            for i in range(feature_cols):
                stat, p = shapiro(X_train_scaled[:, i])
                if p < 0.05:
                    non_normal_features.append(i)
            print(f"\n{len(non_normal_features)} out of {feature_cols} features are NOT normally distributed.")

            # Calculate correlation of each feature with age (Spearman correlation)
            feature_correlations = np.array([spearmanr(X_train_scaled[:, i], y_train)[0] for i in range(X_train_scaled.shape[1])])
            
            # Get the indices of the top 10 features most correlated with age (absolute correlation)
            top_10_features_indices = np.argsort(np.abs(feature_correlations))[-10:]
            X_train_scaled = X_train_scaled[:, top_10_features_indices]
            X_test_scaled = X_test_scaled[:, top_10_features_indices]
            print(f"Top 10 features selected based on correlation with age: {top_10_features_indices}")
            print(f"Feature correlations with age: {feature_correlations[top_10_features_indices]}")

        # Train the model
        model.fit(X_train_scaled, y_train)
        model_predictions_for_fold[test_index] = model.predict(X_test_scaled)

    # Evaluate the model after training
    evaluate_model(model_name, ages, model_predictions_for_fold)

# %%
# Save results to a CSV file
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = 'results_upd'
os.makedirs(output_dir, exist_ok=True)
results_df = pd.DataFrame(results)
results_filename = os.path.join(output_dir, f'results_{features_set}_{current_time}.csv')
results_df.to_csv(results_filename, index=False)
print(f"Results saved to {results_filename}")

# %%
# Identify best model based on MAE (lowest)
predictions_dir = 'predictions_upd'
os.makedirs(predictions_dir, exist_ok=True)

best_mae_row = results_df.loc[results_df['MAE'].idxmin()]
best_mae_model = best_mae_row['model']
print(f"Best model based on MAE: {best_mae_model} (MAE = {best_mae_row['MAE']:.4f})")

best_mae_predictions = model_predictions[best_mae_model]
best_mae_df = pd.DataFrame({
    'predicted_age': best_mae_predictions,
    'true_age': ages
})
best_mae_filename = os.path.join(predictions_dir, f'{best_mae_model.replace(" ", "_")}_{features_set}_{current_time}.csv')
best_mae_df.to_csv(best_mae_filename, index=False)
print(f"Best MAE model predictions saved to {best_mae_filename}")

# Identify best model based on Pearson Correlation (highest)
best_pearson_row = results_df.loc[results_df['Pearson'].idxmax()]
best_pearson_model = best_pearson_row['model']
print(f"Best model by Pearson correlation: {best_pearson_model} (r = {best_pearson_row['Pearson']:.4f})")

best_pearson_predictions = model_predictions[best_pearson_model]
best_pearson_df = pd.DataFrame({
    'predicted_age': best_pearson_predictions,
    'true_age': ages
})
best_pearson_filename = os.path.join(predictions_dir, f'{best_pearson_model.replace(" ", "_")}_{features_set}_{current_time}.csv')
best_pearson_df.to_csv(best_pearson_filename, index=False)
print(f"Best Pearson model predictions saved to {best_pearson_filename}")