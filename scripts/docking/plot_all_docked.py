# %%

import pandas as pd
from pathlib import Path
import tarfile
import gzip
from glob import glob
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJ_DIR = Path(__file__).parent.parent.parent
print(PROJ_DIR)
# %%

score_ls = []
molid_ls = []

# %%

directory = Path(f"{PROJ_DIR}/docking/PyMolGen/")
tar_files = [file.name for file in directory.glob("*tar.gz")]
#%%

for tar_file in tar_files:
    molid = tar_file[:-7]
    moldir = f"{directory}/{tar_file}"

    with tarfile.open(moldir) as tar:
        try:
            score_file = tar.getmember(f"{molid}/{molid}_all_scores.csv.gz")
            with tar.extractfile(score_file) as file:
                with gzip.open(file, "rt", encoding="utf-8") as unzipped_file:
                    mol_df = pd.read_csv(unzipped_file).sort_values(by='Affinity(kcal/mol)')

                    score = mol_df.iloc[0]['Affinity(kcal/mol)']
                    score_ls.append(score)
                    molid_ls.append(molid)

        except Exception as e:
            print(e)

            try:
                score_file = tar.getmember(f"{molid}/{molid}_all_scores.csv")
                with tar.extractfile(score_file) as file:
                    mol_df = pd.read_csv(io.BytesIO(file.read()))
                    score = mol_df.iloc[0]['Affinity(kcal/mol)']
                    score_ls.append(score)
                    molid_ls.append(molid)

            except Exception as e:
                print(e)
        

# %%
df = pd.DataFrame(index=molid_ls)
df['Affinity(kcal/mol)'] = score_ls
df = df.sort_values(by="Affinity(kcal/mol)", ascending=True)
# %%
df.to_csv(f"{directory}/all_docking_scores.csv", index_label='ID')


# %%
df = pd.read_csv(f"{directory}/all_docking_scores.csv")
df = df.sort_values(by="Affinity(kcal/mol)", ascending=True)

# Before plotting, ensure the column is numeric
df['Affinity(kcal/mol)'] = pd.to_numeric(df['Affinity(kcal/mol)'], errors='coerce')

# Drop any NaN values that might have been created
df = df.dropna(subset=['Affinity(kcal/mol)'])

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index, y=df['Affinity(kcal/mol)'])

# Check if we have valid data before calculating ticks
if not df.empty and pd.api.types.is_numeric_dtype(df['Affinity(kcal/mol)']):
    # Set y-axis ticks in 0.5 increments
    y_min = np.floor(float(df['Affinity(kcal/mol)'].min()) * 2) / 2  # Explicitly convert to float
    y_max = np.ceil(float(df['Affinity(kcal/mol)'].max()) * 2) / 2
    
    # Generate y-ticks in 0.5 increments
    y_ticks = np.arange(y_min, y_max + 0.5, 0.5)
    plt.yticks(y_ticks)

# Add labels and title
plt.title('Docking Affinity Plot')
plt.xlabel('Index')
plt.ylabel('Affinity (kcal/mol)')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# Load your data
df = pd.read_csv(f"{directory}/all_docking_scores.csv")

# Before plotting, ensure the column is numeric
df['Affinity(kcal/mol)'] = pd.to_numeric(df['Affinity(kcal/mol)'], errors='coerce')

# Drop any NaN values that might have been created
df = df.dropna(subset=['Affinity(kcal/mol)'])

# Sort the dataframe by affinity values (most negative first, as better scores are more negative)
df_sorted = df.sort_values('Affinity(kcal/mol)', ascending=True)

# Calculate percentile thresholds (for negative values, lower/more negative = better)
percentiles = {
    "top_2.5%": np.percentile(df['Affinity(kcal/mol)'], 2.5),
    "top_5%": np.percentile(df['Affinity(kcal/mol)'], 5),
    "top_10%": np.percentile(df['Affinity(kcal/mol)'], 10),
    "top_25%": np.percentile(df['Affinity(kcal/mol)'], 25),
    "top_50%": np.percentile(df['Affinity(kcal/mol)'], 50)
}

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data
sns.lineplot(data=df, x=df.index, y='Affinity(kcal/mol)', ax=ax)

# Set y-axis ticks in 0.5 increments
y_min = np.floor(float(df['Affinity(kcal/mol)'].min()) * 2) / 2
y_max = np.ceil(float(df['Affinity(kcal/mol)'].max()) * 2) / 2
y_ticks = np.arange(y_max, y_min - 0.1, -0.5)
ax.set_yticks(y_ticks)

# Get the plot limits
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Define colors for different percentile regions
colors = {
    "top_2.5%": "#FF9999",  # Light red
    "top_5%": "#FFCC99",    # Light orange
    "top_10%": "#FFFF99",   # Light yellow
    "top_25%": "#CCFF99",   # Light green
    "top_50%": "#99CCFF"    # Light blue
}

# Add shading for each percentile region
# Start with the smallest region (top 2.5%)
ax.add_patch(Rectangle((x_min, y_min), width=x_max-x_min, 
                       height=percentiles["top_2.5%"]-y_min, 
                       color=colors["top_2.5%"], alpha=0.3,
                       label="Top 2.5%"))

# Add the region between top 2.5% and top 5%
ax.add_patch(Rectangle((x_min, percentiles["top_2.5%"]), width=x_max-x_min, 
                       height=percentiles["top_5%"]-percentiles["top_2.5%"], 
                       color=colors["top_5%"], alpha=0.3,
                       label="Top 5%"))

# Add the region between top 5% and top 10%
ax.add_patch(Rectangle((x_min, percentiles["top_5%"]), width=x_max-x_min, 
                       height=percentiles["top_10%"]-percentiles["top_5%"], 
                       color=colors["top_10%"], alpha=0.3,
                       label="Top 10%"))

# Add the region between top 10% and top 25%
ax.add_patch(Rectangle((x_min, percentiles["top_10%"]), width=x_max-x_min, 
                       height=percentiles["top_25%"]-percentiles["top_10%"], 
                       color=colors["top_25%"], alpha=0.3,
                       label="Top 25%"))

# Add the region between top 25% and top 50%
ax.add_patch(Rectangle((x_min, percentiles["top_25%"]), width=x_max-x_min, 
                       height=percentiles["top_50%"]-percentiles["top_25%"], 
                       color=colors["top_50%"], alpha=0.3,
                       label="Top 50%"))

# Add horizontal lines at each percentile boundary
for percentile, value in percentiles.items():
    plt.axhline(y=value, color='gray', linestyle='--', alpha=0.7)
    plt.text(x_max + 0.01 * (x_max - x_min), value, 
             f"{percentile} ({value:.2f})", va='center')

# Add labels and title
plt.title('Docking Affinity Plot Distribution')
plt.xlabel('Number of Molecules')
plt.ylabel('Affinity (kcal/mol)')

# Add legend
plt.legend(loc='upper left')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

#%%
plt.savefig(f"{directory}/docking_score_distribution.png", dpi=600)

 # %%
