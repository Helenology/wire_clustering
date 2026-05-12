import os
import json
import time
import numpy as np
import pandas as pd

# sklearn
from sklearn.cluster import DBSCAN as sklearnDBSCAN

# cuML-related libraries
import cudf
import cuml
import cupy as cp
from cuml.cluster import DBSCAN as cumlDBSCAN
print(cuml.__version__)

# metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------
# Load Experiment Parameters from config.json
# ---------------------------------------------------------
with open(os.path.join("data", "config.json"), "r") as f:
    config = json.load(f)
n_samples    = config["n_samples"]
n_features   = config["n_features"]
n_clusters   = config["n_clusters"]
n_replicates = config["n_replicates"]

# DBSCAN hyperparameters
# For 500-dimensional Gaussian blobs with cluster_std in [2, 10],
# within-cluster Euclidean distances scale as ~ std * sqrt(n_features).
# The tightest cluster (std=2) has mean pairwise distance ~ 2*sqrt(500) ≈ 63.
# We set eps to capture dense neighborhoods within the tightest clusters.
dbscan_eps = 75.0
dbscan_min_samples = 10

# Lists to store the metrics for each replicate
cpu_runtimes = []
gpu_runtimes = []
cpu_aris = []
gpu_aris = []
cpu_nmis = []
gpu_nmis = []

print(f"Starting DBSCAN Benchmark with {n_replicates} replicates...")
print(f"Dataset per replicate: {n_samples} samples, {n_features} features, {n_clusters} clusters")
print(f"DBSCAN params: eps={dbscan_eps}, min_samples={dbscan_min_samples}")

# =========================================================
# WARM-UP PHASE
# =========================================================
print("\n--- Performing GPU & CPU Warm-up ---")
# Load a small slice of the first replicate for warming up the CUDA context
warm_data = np.load(os.path.join("data", "replicate_0.npz"))
X_warm_cpu = warm_data["X"][:1000].astype(np.float32)
X_warm_gpu = cp.asarray(X_warm_cpu)

# Run a dummy training on the CPU and GPU to load libraries into cache
_ = sklearnDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_warm_cpu)
_ = cumlDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_warm_gpu)
print("Warm-up complete. The actual benchmark will now begin.")

# =========================================================
# MAIN EXPERIMENT LOOP
# =========================================================

for i in range(n_replicates):
    print(f"\n--- Running Replicate {i + 1}/{n_replicates} ---")

    # Load the pre-generated dataset from disk
    data = np.load(os.path.join("data", f"replicate_{i}.npz"))
    X = data["X"]
    y_true = data["y"]

    # Convert and transfer data
    X_cpu = X.astype(np.float32)
    X_gpu = cp.asarray(X_cpu)

    # 1. Scikit-learn (CPU)
    start_time = time.time()
    sk_dbscan = sklearnDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    sk_labels = sk_dbscan.fit_predict(X_cpu)
    cpu_time = time.time() - start_time
    cpu_runtimes.append(cpu_time)
    cpu_aris.append(adjusted_rand_score(y_true, sk_labels))
    cpu_nmis.append(normalized_mutual_info_score(y_true, sk_labels))

    # 2. cuML (GPU)
    start_time = time.time()
    cu_dbscan = cumlDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    cu_labels = cu_dbscan.fit_predict(X_gpu)
    gpu_time = time.time() - start_time
    gpu_runtimes.append(gpu_time)
    cu_labels_cpu = cu_labels.get()
    gpu_aris.append(adjusted_rand_score(y_true, cu_labels_cpu))
    gpu_nmis.append(normalized_mutual_info_score(y_true, cu_labels_cpu))

    # =========================================================
    # SAVE RESULTS TO DISK (After each replicate for robustness)
    # =========================================================

    # Create a dictionary to organize the lists of metrics we collected during the loop
    results_dict = {
        # Generate a sequence of numbers from 1 to n_replicates to track the iteration
        "Replicate": list(range(1, i + 2)),
        # Add the Scikit-learn CPU execution times
        "CPU_Runtime_sec": cpu_runtimes,
        # Add the cuML GPU execution times
        "GPU_Runtime_sec": gpu_runtimes,
        # Add the Scikit-learn CPU Adjusted Rand Index scores
        "CPU_ARI": cpu_aris,
        # Add the cuML GPU Adjusted Rand Index scores
        "GPU_ARI": gpu_aris,
        "CPU_NMI": cpu_nmis,
        "GPU_NMI": gpu_nmis
    }

    # Convert the organized dictionary into a pandas DataFrame structure
    results_df = pd.DataFrame(results_dict)

    # Define the filename where the results will be saved
    output_filename = "python_dbscan_results.csv"

    # Save the DataFrame to a CSV file, setting index=False to prevent writing row numbers
    results_df.to_csv(output_filename, index=False)

# ---------------------------------------------------------
# Final Statistics & Benchmark Results Summary
# ---------------------------------------------------------
# Calculate mean and standard deviation for runtimes
cpu_runtime_mean, cpu_runtime_std = np.mean(cpu_runtimes), np.std(cpu_runtimes)
gpu_runtime_mean, gpu_runtime_std = np.mean(gpu_runtimes), np.std(gpu_runtimes)

# Calculate mean and standard deviation for ARI
cpu_ari_mean, cpu_ari_std = np.mean(cpu_aris), np.std(cpu_aris)
gpu_ari_mean, gpu_ari_std = np.mean(gpu_aris), np.std(gpu_aris)

# Calculate mean and standard deviation for NMI
cpu_nmi_mean, cpu_nmi_std = np.mean(cpu_nmis), np.std(cpu_nmis)
gpu_nmi_mean, gpu_nmi_std = np.mean(gpu_nmis), np.std(gpu_nmis)

# Calculate overall speedup based on mean runtimes
mean_speedup = cpu_runtime_mean / gpu_runtime_mean

print("\n" + "="*50)
print(f"    DBSCAN Benchmark Summary ({n_replicates} Replicates)    ")
print("-" * 50)
print("Runtime (Seconds):")
print(f"  CPU (sklearn):  {cpu_runtime_mean:.4f} ± {cpu_runtime_std:.4f}")
print(f"  GPU (cuML):     {gpu_runtime_mean:.4f} ± {gpu_runtime_std:.4f}")
print(f"  Mean Speedup:   {mean_speedup:.2f}x faster")
print("-" * 50)
print("Adjusted Rand Index (ARI):")
print(f"  CPU (sklearn):  {cpu_ari_mean:.6f} ± {cpu_ari_std:.6f}")
print(f"  GPU (cuML):     {gpu_ari_mean:.6f} ± {gpu_ari_std:.6f}")
print(f"  Mean Diff:      {abs(cpu_ari_mean - gpu_ari_mean):.6e}")
print("-" * 50)
print("Normalized Mutual Information (NMI):")
print(f"  CPU (sklearn):  {cpu_nmi_mean:.6f} ± {cpu_nmi_std:.6f}")
print(f"  GPU (cuML):     {gpu_nmi_mean:.6f} ± {gpu_nmi_std:.6f}")
print(f"  Mean Diff:      {abs(cpu_nmi_mean - gpu_nmi_mean):.6e}")
print("="*50)
