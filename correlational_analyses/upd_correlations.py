# %%
# Import
import pandas as pd
import mne
import os
import re
from scipy.stats import shapiro
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import datetime

# %%
# Directory for upd data
directory = '../../data/upd_data'

# %%
# Read excel with subject descriptions
descriptions = pd.read_excel('../../data/upd_data/vars_of_interest.xlsx', engine='openpyxl')

# keep only relevant columns
columns_to_keep = ['ID', 'age', 'gender']
age_descriptions = descriptions[columns_to_keep]
age_descriptions.set_index('ID', inplace=True)

# %%
# Group by sleep stage to get the total duration of each stage for a single night
def calculate_stage_percentage(annotations_df):
    total_duration = annotations_df['duration'].sum()  # Total sleep time for that night
    stage_durations = annotations_df.groupby('description')['duration'].sum().reset_index()

    # Calculate percentage of time spent in each stage
    stage_durations['percentage'] = (stage_durations['duration'] / total_duration) 
    return stage_durations, total_duration

# %%
# Remove all waking periods before and after sleep (keeping only WASO)
def remove_wake(annotations_df):
    non_waking_indices = annotations_df[annotations_df['description'] != 'W'].index

    first_non_waking_idx = non_waking_indices[0]
    last_non_waking_idx = non_waking_indices[-1]

    cleaned_df = annotations_df.loc[first_non_waking_idx:last_non_waking_idx]

    return cleaned_df

# %%
# Function to extract subject ID from the edf file path
def extract_subject_id(file_path):
    # extract subject number from file name
    # Use regular expression to find the subject number (pattern: digits and dashes)
    subject_number = re.search(r'\d{2}-\d{2}-\d{3}', file_path).group()

    if subject_number:
        return subject_number # return subject ID as integer
    
    return None

# %%
# Extract sleep stages and total sleep time for each subject

# Initialize dictionaries to store sleep stages and total sleep time
subject_sleep_stages = {}
subject_sleep_total = {}
number_psgs = 0

# Loop through three subdirectories (19-01, 21-03, 21-04)
for subdir in os.listdir(directory):

    subdir_path = os.path.join(directory, subdir)

    if os.path.isdir(subdir_path):

        for file in os.listdir(subdir_path):
            
            # Loop through all hypnogram/sleep files in the directory
            if file.endswith('.set'):

                number_psgs += 1

                file_path = os.path.join(subdir_path, file)
                subject_id = extract_subject_id(file_path)

                raw = mne.io.read_raw_eeglab(file_path)
                
                annotations_df = pd.DataFrame({
                    'onset': raw.annotations.onset,
                    'duration': raw.annotations.duration,
                    'description': raw.annotations.description
                })

                sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
                hypnogram_df = annotations_df[annotations_df['description'].isin(sleep_stages)]
                hypnogram_df['duration'] = hypnogram_df['onset'].diff().shift(-1)
                hypnogram_df['duration'].fillna(0, inplace=True)

                # Reset the index of the DataFrame
                hypnogram_df = hypnogram_df.reset_index(drop=True)

                hypnogram_df = remove_wake(hypnogram_df)
                stage_durations, total_duration = calculate_stage_percentage(hypnogram_df)

                subject_sleep_stages[subject_id] = stage_durations
                subject_sleep_total[subject_id] = total_duration

       
# %%
# Convert the sleep stages dictionary to a DataFrame
dfs = []

for subject_id, df in subject_sleep_stages.items():
    df = df.transpose()

    df_reset = df.reset_index()

    df_percentage = df_reset[df_reset['index'] == 'percentage']
    df_duration = df_reset[df_reset['index'] == 'duration']

    df_percentage.columns = df_reset.iloc[0, :].values
    df_duration.columns = df_reset.iloc[0, :].values

    df_percentage.columns = [f'{col}_percentage' for col in df_percentage.columns]
    df_duration.columns = [f'{col}_duration' for col in df_duration.columns]

    df_percentage['subject_id'] = subject_id
    df_duration['subject_id'] = subject_id

    df_percentage.set_index('subject_id', inplace=True)
    df_duration.set_index('subject_id', inplace=True)

    # combine df_percentage and df_duration
    stage_df = pd.concat([df_percentage, df_duration], axis=1)

    dfs.append(stage_df)

# %%
# Combine all the individual DataFrames into one big DataFrame
combined_df = pd.concat(dfs)

# %%
# Fill NaN values with 0 (or another value if needed)
combined_df.fillna(0, inplace=True)

# %%
# add total sleep time to dataframe
subject_sleep_total_df = pd.DataFrame.from_dict(subject_sleep_total, orient='index', columns=['total_sleep_duration'])
combined_df = combined_df.reset_index().merge(subject_sleep_total_df, left_on='subject_id',right_index=True)
combined_df.set_index('subject_id', inplace=True)

# %%
stages_df = combined_df.drop(['description_percentage', 'description_duration'], axis=1)

# %%
# Merge stage durations with age data
merged_df = pd.merge(age_descriptions, stages_df, left_index=True, right_index=True)

# %%
# select and reorder columns
new_order_selection = ['age', 'total_sleep_duration', 'W_duration', 'W_percentage', 'N1_duration', 'N1_percentage', 'N2_duration', 'N2_percentage', 'N3_duration', 'N3_percentage', 'R_duration', 'R_percentage']
merged_df = merged_df[new_order_selection]

# Rename columns in the DataFrame
column_mapping = {
    'total_sleep_duration': 'Total sleep duration',
    'W_duration': 'Wake duration',
    'W_percentage': 'Wake percentage',
    'N1_duration': 'N1 duration',
    'N1_percentage': 'N1 percentage',
    'N2_duration': 'N2 duration',
    'N2_percentage': 'N2 percentage',
    'N3_duration': 'N3 duration',
    'N3_percentage': 'N3 percentage',
    'R_duration': 'REM duration',
    'R_percentage': 'REM percentage'
}

merged_df.rename(columns=column_mapping, inplace=True)
# %%
# Testing normality using Shapiro Wilk
shapiro_results = []
for label in merged_df.columns:
    # Perform the Shapiro-Wilk test for normality
    stat, p_value = shapiro(merged_df[label])
    shapiro_results.append({'feature': label, 'statistic': stat, 'p_value': p_value})

    # Check the p-value
    if p_value > 0.05:
        print(f"{label} is normally distributed (fail to reject H0)")
    else:
        print(f"{label} is not normally distributed (reject H0)")

    print(f"Shapiro-Wilk test statistic for {label}: {stat}, p-value: {p_value}")

# Convert results to DataFrame
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.set_index('feature', inplace=True)

# Save to CSV
dir = 'results/shapiro'
os.makedirs(dir, exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
shapiro_df.to_csv(os.path.join(dir, f'{current_time}_shapiro_results_upd.csv'))

# %%
# Perform Spearman correlation for all columns
correlations = []
p_values = []

features = merged_df.columns
features = features.drop('age')
age = merged_df['age']  

# Compute Spearman correlation and p-values
for feature in features:
    rho, p = spearmanr(merged_df[feature], age)
    correlations.append(rho)
    p_values.append(p)

# Convert to DataFrame
results = pd.DataFrame({'feature': features, 'Spearman_rho': correlations, 'p_value': p_values})

# Multiple testing correction
_, p_bonferroni, _, _ = multipletests(p_values, method='bonferroni')
results['p_bonferroni'] = p_bonferroni
results['significant'] = results['p_bonferroni'] < 0.05

# Display results
print(results)

# Save results to CSV
dir = 'results/correlations'
os.makedirs(dir, exist_ok=True) 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
results.to_csv(os.path.join(dir, f'{current_time}_spearman_correlations_upd.csv'))
merged_df.to_csv(os.path.join(dir, f'{current_time}_merged_df_upd.csv'))
