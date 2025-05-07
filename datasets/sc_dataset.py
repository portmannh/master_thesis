# %%
'''
Statistical description of SC dataset
Results are saved in a CSV file
'''

# %%
# Imports
import pandas as pd
import os

# %%
# Read excel
descriptions = pd.read_excel('../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls')

# %%
# Get row count (number of PSGs)
row_count_psg = descriptions.shape[0]

# %%
# Drop duplicate subjects (keep only one row per subject)
descriptions = descriptions.drop_duplicates(subset=['subject'], keep='first')

# %%
# Get row count (number of subjects)
row_count_subjects = descriptions.shape[0]

# %%
# Get age statistics
min_age = descriptions['age'].min()
max_age = descriptions['age'].max()
mean_age = descriptions['age'].mean()
std_age = descriptions['age'].std()

# %%
# Get sex statistics
sex_counts = descriptions['sex (F=1)'].value_counts()
n_female = sex_counts.get(1, 0)
n_male = sex_counts.get(2, 0)

# %%
# Create a DataFrame with results
result_data = {
    'Metric': ['Number of PSGs', 'Number of subjects', 'Age - Min', 'Age - Max', 'Age - Mean', 'Age - Std', 'Number of female subjects', 'Number of male subjects'],
    'Value': [row_count_psg, row_count_subjects, min_age, max_age, mean_age, std_age, n_female, n_male]
}

results_df = pd.DataFrame(result_data)

# Save to CSV
# Specify the path where you want to save the CSV file
path = '/home/hp22w468/Master/master_thesis_project/master_thesis_scripts/datasets/results'

# create directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

results_df.to_csv(os.path.join(path, 'sc_desctiptions'), index=False)


