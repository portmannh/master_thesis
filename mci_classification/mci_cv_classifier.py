"""
MCI Classification of UPD set, using LOO CV
Using various feature input types and classifiers
"""

# %% 
# imports
import pandas as pd
import os
import datetime
import numpy as np
import tensorflow as tf
import random

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

from sklearn.decomposition import PCA

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
# load data
features_dir = '../age_regression/features_upd'

# Select type of features to use
features_set = 'averaged'  # options: 'averaged', 'flattened', 'averaged_pca', 'flattened_pca'
pca_used = 'pca' in features_set

data_load_averaged = 'averaged' in features_set
if data_load_averaged:
    data_name = 'averaged'
else:
    data_name = 'flattened'

# %%
# Load data
healthy_name = f'healthy_data_{data_name}.csv'
mci_name = f'mci_data_{data_name}.csv'
healthy_df = pd.read_csv(os.path.join(features_dir, healthy_name))
mci_df = pd.read_csv(os.path.join(features_dir, mci_name))

# Adding MCI column
healthy_df['mci'] = 0
mci_df['mci'] = 1

if pca_used:
    data_name = f"{data_name}_pca"

# %%
# Put both dataframes together (can be differentiated by mci column)
data = pd.concat([healthy_df, mci_df], ignore_index=True)
data = data.sample(frac=1, random_state=seed)  # Shuffle the data
data = data.drop(columns=["age", "moca"]) # drop age and moca columns

# %%
# Define classifiers and hyperparameters
# Compute class priors
prior_healthy = len(healthy_df) / len(data)
prior_mci = len(mci_df) / len(data)

# Define all classifiers
# Note: The class_weight='balanced' parameter and the priors are used to handle class imbalance
classifiers = {
    "Logistic Regression": [LogisticRegression(class_weight='balanced', max_iter=1000)],
    "SVM": [SVC(probability=True, kernel='linear', class_weight='balanced'), SVC(probability=True, kernel='rbf', class_weight='balanced')],
    "LDA": [LinearDiscriminantAnalysis(priors=[prior_healthy, prior_mci])],
    "Naive Bayes": [NB(priors=[prior_healthy, prior_mci])],
    "Random Forest": [RandomForestClassifier(n_estimators=100, class_weight='balanced')]
}
# %%
# LOO CV
results = []
loo = LeaveOneOut()

# %%
# Loop over all models
for model_name, model_variants in classifiers.items():
    for model in model_variants:
        y_true, y_scores = [], []

        for train_index, test_index in loo.split(data):
            X_train, X_test = data.iloc[train_index].drop(columns=["mci"]), data.iloc[test_index].drop(columns=["mci"])
            y_train, y_test = data.iloc[train_index]["mci"], data.iloc[test_index]["mci"]
            
            # Scale data using StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Apply PCA if needed
            if pca_used:
                pca = PCA(n_components=10)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                # Check explained variance ratio
                print("Explained Variance Ratio:", pca.explained_variance_ratio_)
                print("Total Variance Explained:", np.sum(pca.explained_variance_ratio_))
            
            # Fit the model
            model.fit(X_train, y_train)
            # Predict probabilities              
            prob = model.predict_proba(X_test)[:, 1]

            y_true.append(y_test.values[0])
            y_scores.append(prob[0])
        
        # Compute AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy
        # Convert probabilities to binary predictions
        y_pred = [1 if score > 0.5 else 0 for score in y_scores]
        miscl = sum(t != p for t, p in zip(y_true, y_pred))
        accuracy = 1 - miscl/len(y_true)

        # Compute F1 score
        f1 = f1_score(y_true, y_pred)

        results.append({
            "Model": model_name,
            "Hyperparameters": str(model.get_params()),
            "AUC": roc_auc,
            "Accuracy": accuracy,
            "F1": f1,
            "FPR": np.array2string(fpr, separator=','),
            "TPR": np.array2string(tpr, separator=','),
            "y_true": str(y_true),
            "y_pred": str(y_pred)
        })


# %%
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Determine best models
best_auc = results_df[results_df['AUC'] == results_df['AUC'].max()]
best_f1 = results_df[results_df['F1'] == results_df['F1'].max()]

print("Best Models by AUC:")
print(best_auc[['Model', 'Hyperparameters', 'AUC', 'Accuracy', 'F1']])
print("\nBest Models by F1:")
print(best_f1[['Model', 'Hyperparameters', 'AUC', 'Accuracy', 'F1']])

# %%
# Save results
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(os.path.join(results_dir, f"cv_classifier_results_{features_set}_{current_time}.csv"))

best_auc.to_csv(os.path.join(results_dir, f"best_auc_models_{features_set}_{current_time}.csv"))
best_f1.to_csv(os.path.join(results_dir, f"best_f1_models_{features_set}_{current_time}.csv"))
