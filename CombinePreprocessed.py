import os
import glob
import pandas as pd
DATASET_DIR = "MIT_Preprocessed_Datasets"
# Get all CSV files
csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"Found {len(csv_files)} files.")

# Read every CSV
dfs = [pd.read_csv(file) for file in csv_files]

# Combine them
combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df.shape)

# Save
combined_df.to_csv(
    "MITBIH_combined.csv",
    index=False
)
print("Combined dataset saved.")