# %%
'''
Statistical description of UPD dataset
Results are saved in a CSV file
'''

# %%
# Imports
import pandas as pd
import os

# %%
# Read excel
descriptions = pd.read_excel('../../data/upd_data/vars_of_interest.xlsx', engine='openpyxl')

# %%
# Get rid of unnecessary columns
columns_to_keep = ['ID', 'original_Group', 'gender', 'age', 'MOCA_score']
descriptions_upd = descriptions[columns_to_keep]

# %%
# Create DataFrames for each study
groups = descriptions_upd['original_Group'].unique()

# Create a dictionary to hold the individual DataFrames
group_dfs = {group: descriptions_upd[descriptions_upd['original_Group'] == group] for group in groups}

df_1901 = group_dfs[groups[0]]
df_2103 = group_dfs[groups[1]]
df_2104 = group_dfs[groups[2]]

# %%
summary_stats = {}

# Calculate summary statistics for each group
for group, df in group_dfs.items():
    summary_stats[group] = {
        'mean_age': df['age'].mean(),
        'std_age': df['age'].std(),
        'min_age': df['age'].min(),
        'max_age': df['age'].max(),
        'n_male': df['gender'].value_counts()[1],
        'n_female': df['gender'].value_counts()[0],
        'n_healthy': (df['MOCA_score']>=26).sum(),
        'n_mci': (df['MOCA_score']<26).sum(),
        'mean_moca': df['MOCA_score'].mean(),
        'std_moca': df['MOCA_score'].std(),
    }

# %%
# Calculate summary statistics for the entire dataset
summary_stats['upd'] = {
        'mean_age': descriptions_upd['age'].mean(),
        'std_age': descriptions_upd['age'].std(),
        'min_age': descriptions_upd['age'].min(),
        'max_age': descriptions_upd['age'].max(),
        'n_male': descriptions_upd['gender'].value_counts()[1],
        'n_female': descriptions_upd['gender'].value_counts()[0],
        'n_healthy': (descriptions_upd['MOCA_score']>=26).sum(),
        'n_mci': (descriptions_upd['MOCA_score']<26).sum(),
        'mean_moca': descriptions_upd['MOCA_score'].mean(),
        'std_moca': descriptions_upd['MOCA_score'].std(),
    }

# %%
# Divide dataset into MCI and healthy groups
healthy_df = descriptions_upd[descriptions_upd['MOCA_score']>=26]
mci_df = descriptions_upd[descriptions_upd['MOCA_score']<26]

# %%
# Calculate summary statistics for MCI and healthy groups
dfs = {'mci': mci_df, 'healthy': healthy_df}
for name, df in dfs.items():
    summary_stats[name] = {
        'mean_age': df['age'].mean(),
        'std_age': df['age'].std(),
        'min_age': df['age'].min(),
        'max_age': df['age'].max(),
        'n_male': df['gender'].value_counts()[1],
        'n_female': df['gender'].value_counts()[0],
        'n_healthy': (df['MOCA_score']>=26).sum(),
        'n_mci': (df['MOCA_score']<26).sum(),
        'mean_moca': df['MOCA_score'].mean(),
        'std_moca': df['MOCA_score'].std(),
    }



# %%
# Create a DataFrame with results
# Transpose so that metrics are columns and groups are rows
results_df = pd.DataFrame(summary_stats).T.reset_index()
results_df.rename(columns={'index': 'group'}, inplace=True)

# %%
# Save to CSV
# Specify the path where you want to save the CSV file
path = '/home/hp22w468/Master/master_thesis_project/master_thesis_scripts/datasets/results'

# create directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

results_df.to_csv(os.path.join(path, 'upd_desctiptions'), index=False)

