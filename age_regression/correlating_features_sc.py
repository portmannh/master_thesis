'''
Correlating the extracted averaged SC features with age and selecting only the 10 most correlated features
Features are selected based on correlation with age in training set
'''

# %% 
# imports
import pandas as pd
import os
import numpy as np
import random
import tensorflow as tf
from scipy.stats import shapiro

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
# load data
features_dir = f'features_sc'

# Load the DataFrame
train_df = pd.read_csv(os.path.join(features_dir, "train_data_averaged.csv"))
val_df = pd.read_csv(os.path.join(features_dir, "val_data_averaged.csv"))
test_df = pd.read_csv(os.path.join(features_dir, "test_data_averaged.csv"))

# Check normality of age
stat, p_age = shapiro(train_df['age'])
print(f"Shapiro-Wilk test for 'age': p-value = {p_age:.4f} -> {'Normal' if p_age > 0.05 else 'Not normal'}")

# Check normality of averaged features
feature_cols = train_df.columns.drop('age')
non_normal_features = []

for feat in feature_cols:
    stat, p = shapiro(train_df[feat])
    if p < 0.05:
        non_normal_features.append(feat)

print(f"\n{len(non_normal_features)} out of {len(feature_cols)} features are NOT normally distributed.")


# %%
## Calculate correlations with age
correlations = train_df.corr(method='spearman')

# Extract the correlation values between 'age' and all other features
age_correlations = correlations['age'].drop('age')  # Exclude 'age' itself

# Sort correlations by absolute value in descending order
age_correlations = age_correlations.abs().sort_values(ascending=False)

# Print top 10 features with their correlation values
print("Top 10 features most correlated with age:")
print(age_correlations.head(10))

# %%
# Select the indices of the top 10 correlated features (in train set)
top_10_indices = age_correlations.index[:10]

# %%
# keep only the top 10 correlated features in train, val and test sets
top_10_train = train_df[top_10_indices].copy()
top_10_train['age'] = train_df['age']
top_10_val = val_df[top_10_indices].copy()
top_10_val['age'] = val_df['age']
top_10_test = test_df[top_10_indices].copy()
top_10_test['age'] = test_df['age']

# %%
# Save the selected features and ages to CSV files
top_10_train.to_csv(os.path.join(features_dir, 'train_data_correlated.csv'), index=False)
top_10_val.to_csv(os.path.join(features_dir, 'val_data_correlated.csv'), index=False)
top_10_test.to_csv(os.path.join(features_dir, 'test_data_correlated.csv'), index=False)

print(f'Top 10 correlated Feature vectors and ages saved to {features_dir}')