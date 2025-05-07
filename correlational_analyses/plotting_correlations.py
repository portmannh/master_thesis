# %%
# Imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Results path (change depending on data)
results_path = 'results/correlations/20250422-122251_spearman_correlations_upd.csv'
results = pd.read_csv(results_path)
merged_df_path = 'results/correlations/20250422-122251_merged_df_upd.csv'
merged_df = pd.read_csv(merged_df_path)

# %%
# Set the figure directory
fig_dir = 'figures/upd' # change to data directory
os.makedirs(fig_dir, exist_ok=True)

# %%
# Function to plot correlation
def plot_correlation(df, feature, results_df, age_col='age'):

    # get the Spearman rho and p-value for the feature
    row = results_df[results_df['feature'] == feature]
    if row.empty:
        print(f"No data found for feature: {feature}")
        return
    
    rho = row['Spearman_rho'].values[0]
    p_value = row['p_bonferroni'].values[0]
    significant = row['significant'].values[0]

    # Scatterplot
    plt.figure(figsize=(8, 6))
    sns.set_context("paper")
    sns.set_palette("colorblind")

    sns.scatterplot(x=df[age_col], y=df[feature], alpha=0.7, color='black')

    # Add linear regression trend line
    sns.regplot(x=df[age_col], y=df[feature], scatter=False, color='red', ci=None, line_kws={'linewidth': 2})

    # Add the rho and p-value on the plot in a readable position
    x_pos = df[age_col].min() * 1.05  # Slightly shift text from the left
    y_pos = df[feature].max() * 0.95  # Slightly shift text from the top
    plt.text(x=x_pos, y=y_pos, s=f'$\\rho$ = {rho:.2f}, p = {p_value:.3g}', 
             fontsize=14, color='black', ha='left', va='top')

    # add unit to feature name
    if feature.lower().find('percentage') != -1:
        y_label = feature.replace('_', ' ').capitalize() + ' (%)'
    elif feature.lower().find('duration') != -1:
        y_label = feature.replace('_', ' ').capitalize() + ' (seconds)'

    # Labels and title
    plt.title(f'Age vs. {feature}', fontsize=16)
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    plt.tight_layout()

    feature_clean = feature.replace(" ", "_")

    plt.savefig(os.path.join(fig_dir, f'{feature_clean}_correlations.png'), format='png', dpi=300)

    
# %%
# Plot correlation for each sleep stage
sleep_features = results['feature'].tolist()

for feature in sleep_features:
    plot_correlation(merged_df, feature, results)