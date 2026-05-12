"""
Pre-generate all replicate datasets and save to disk.
Run this ONCE before running any benchmark scripts.

Usage:
    python generate_data.py
"""

import os
import json
import numpy as np
from sklearn.datasets import make_blobs


# ---------------------------------------------------------
# Experiment Parameters (SINGLE SOURCE OF TRUTH)
# All benchmark scripts load these from data/config.json
# ---------------------------------------------------------
n_samples    = 10000
n_features   = 10
n_clusters   = 2
n_replicates = 100

# Create the output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Save parameters as config.json for all benchmark scripts to load
config = {
    "n_samples":    n_samples,
    "n_features":   n_features,
    "n_clusters":   n_clusters,
    "n_replicates": n_replicates
}
config_path = os.path.join(output_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved config to {config_path}")

print(f"Generating {n_replicates} datasets...")
print(f"Each dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
print(f"Saving to: {output_dir}/\n")


# Define the data generation function
def generate_dataset(n_samples, n_features, n_clusters, seed=0):
    std_array = np.linspace(start=1, stop=(n_clusters * 2), num=n_clusters)
    
    # Generate the feature matrix X and true labels y_true using make_blobs
    X, y = make_blobs(n_samples=n_samples, 
                           n_features=n_features, 
                           centers=n_clusters, 
                           random_state=seed,
                           cluster_std=std_array,
                           center_box=(-50.0, 50.0)
                          )
    return X, y


for i in range(n_replicates):
    filepath = os.path.join(output_dir, f"replicate_{i}.npz")

    # Generate dataset with seed=i (matches original benchmark convention)
    X, y = generate_dataset(n_samples=n_samples, n_features=n_features,
                                  n_clusters=n_clusters, seed=i)

    # Save as compressed numpy archive
    np.savez_compressed(filepath, X=X, y=y)

    print(f"  Saved replicate {i + 1}/{n_replicates} -> {filepath}")

print(f"\nDone. {n_replicates} datasets saved to {output_dir}/")
