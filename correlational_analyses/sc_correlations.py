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
# Directory for sleep data
directory = '../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'

# %%
# Read excel with subject descriptions
descriptions = pd.read_excel('../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls')

# keep only relevant columns (age of subjects)
descriptions.drop(['night', 'LightsOff'], axis=1, inplace=True)
descriptions.drop_duplicates(['subject'], inplace=True)
descriptions.set_index('subject', inplace=True)

# %%
# Group by sleep stage to get the total duration of each stage for a single night
def calculate_stage_percentage(annotations_df):
    total_duration = annotations_df['duration'].sum()  # Total sleep time for that night
    stage_durations = annotations_df.groupby('sleep_stage')['duration'].sum().reset_index()

    # Calculate percentage of time spent in each stage
    stage_durations['percentage'] = (stage_durations['duration'] / total_duration) 
    return stage_durations, total_duration

# %%
# Revove all waking periods before and after sleep (keeping only WASO)
def remove_wake(annotations_df):
    non_wake = annotations_df[annotations_df['sleep_stage'] != 'Sleep stage W']

    # check if there are any waking periods
    if non_wake.empty:
        return annotations_df
    
    # keep everything from first non_wake to last non-wake
    first_sleep_idx = non_wake.index[0]
    last_sleep_idx = non_wake.index[-1]
    cleaned_df = annotations_df.loc[first_sleep_idx:last_sleep_idx]

    return cleaned_df

# %%
# Function to extract subject ID from the edf file path
def extract_subject_id(file_path):
    # extract subject number from file name
    pattern = r'SC4(\d{2})\d'
    match = re.search(pattern, file_path)

    if match:
        return int(match.group(1)) # return subject ID as integer
    
    return None

# %%
# Extract sleep stages and total sleep time for each subject

# Initialize dictionaries to store sleep stages and total sleep time
subject_sleep_stages = {}
subject_sleep_total = {}
number_psgs = 0

# Loop through all hypnogram files in the directory
for file in os.listdir(directory):

    if file.endswith('Hypnogram.edf'):

        file_path = os.path.join(directory, file)
        subject_id = extract_subject_id(file_path)

        hypnogram = mne.read_annotations(file_path)

        hypnogram_df =  pd.DataFrame({
            'onset': hypnogram.onset,        # Start time of each annotation (in seconds)
            'duration': hypnogram.duration,  # Duration of each annotation (in seconds)
            'sleep_stage': hypnogram.description  # Sleep Stage
        })

        # Modify hypnogram DataFrame to combine sleep stages 3 and 4
        hypnogram_df['sleep_stage'] = hypnogram_df['sleep_stage'].replace({'Sleep stage 4': 'Sleep stage 3'})

        # Remove waking periods before and after sleep
        hypnogram_df = remove_wake(hypnogram_df)

        # Compute durations and percentages for each sleep stage
        stage_durations, total_duration = calculate_stage_percentage(hypnogram_df)

        number_psgs += 1

        # Check if the subject ID already exists in the dictionary (for those subjects with two nights)
        if subject_id in subject_sleep_stages:
            existing_data = subject_sleep_stages[subject_id]
            combined_data = pd.merge(existing_data, stage_durations, on='sleep_stage', suffixes=('_night1', '_night2'))

            # Calculate the average percentage and duration for each sleep stage
            combined_data['percentage'] = (combined_data['percentage_night1'] + combined_data['percentage_night2']) / 2
            combined_data['duration'] = (combined_data['duration_night1'] + combined_data['duration_night2']) / 2
  
            subject_sleep_stages[subject_id] = combined_data[['sleep_stage', 'percentage', 'duration']]

            # Calculate the average total sleep time
            existing_total = subject_sleep_total[subject_id]
            subject_sleep_total[subject_id] = (existing_total + total_duration) / 2

        # If the subject ID does not exist, add it to the dictionary
        else:
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

# Combine all the individual DataFrames into one big DataFrame
combined_df = pd.concat(dfs)

# Fill NaN values with 0 
combined_df.fillna(0, inplace=True)

# %%
# Add total sleep time to dataframe
subject_sleep_total_df = pd.DataFrame.from_dict(subject_sleep_total, orient='index', columns=['total_sleep_duration'])
combined_df = combined_df.reset_index().merge(subject_sleep_total_df, left_on='subject_id',right_index=True)
combined_df.set_index('subject_id', inplace=True)

# %%
stages_df = combined_df.drop(['sleep_stage_percentage', 'sleep_stage_duration'], axis=1)

# %%
# Merge stage durations with age data
merged_df = pd.merge(descriptions, stages_df, left_index=True, right_index=True)

# %%
# select and re-order columns of dataframe
new_order_selection = ['age', 'total_sleep_duration', 'Sleep stage W_duration', 'Sleep stage W_percentage', 'Sleep stage 1_duration', 'Sleep stage 1_percentage', 'Sleep stage 2_duration', 'Sleep stage 2_percentage', 'Sleep stage 3_duration', 'Sleep stage 3_percentage', 'Sleep stage R_duration', 'Sleep stage R_percentage']
merged_df = merged_df[new_order_selection]

# Rename columns in the DataFrame
column_mapping = {
    'total_sleep_duration': 'Total sleep duration',
    'Sleep stage 1_duration': 'N1 duration',
    'Sleep stage 1_percentage': 'N1 percentage',
    'Sleep stage 2_duration': 'N2 duration',
    'Sleep stage 2_percentage': 'N2 percentage',
    'Sleep stage 3_duration': 'N3 duration',
    'Sleep stage 3_percentage': 'N3 percentage',
    'Sleep stage R_duration': 'REM duration',
    'Sleep stage R_percentage': 'REM percentage',
    'Sleep stage W_duration': 'Wake duration',
    'Sleep stage W_percentage': 'Wake percentage'
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
shapiro_df.to_csv(os.path.join(dir, f'{current_time}_shapiro_results_sc.csv'))

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
results.to_csv(os.path.join(dir, f'{current_time}_spearman_correlations_sc.csv'))
merged_df.to_csv(os.path.join(dir, f'{current_time}_merged_df_sc.csv'))

