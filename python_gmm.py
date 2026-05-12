import os
import json
import time
import numpy as np
import pandas as pd

# sklearn
from sklearn.mixture import GaussianMixture as sklearnGMM

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

# Lists to store the metrics for each replicate
cpu_runtimes = []
cpu_aris = []
cpu_nmis = []

print(f"Starting GMM Benchmark with {n_replicates} replicates...")
print(f"Dataset per replicate: {n_samples} samples, {n_features} features, {n_clusters} clusters")
print("NOTE: cuML does not support GMM — running sklearn (CPU) only.\n")

# =========================================================
# WARM-UP PHASE
# =========================================================
print("--- Performing CPU Warm-up ---")
warm_data = np.load(os.path.join("data", "replicate_0.npz"))
X_warm_cpu = warm_data["X"][:1000].astype(np.float32)
_ = sklearnGMM(n_components=n_clusters, covariance_type='spherical', n_init=1, random_state=0).fit_predict(X_warm_cpu)
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

    X_cpu = X.astype(np.float32)

    # Scikit-learn (CPU)
    start_time = time.time()
    sk_gmm = sklearnGMM(n_components=n_clusters, covariance_type='spherical', n_init=1, max_iter=100, random_state=i)
    sk_labels = sk_gmm.fit_predict(X_cpu)
    cpu_time = time.time() - start_time
    cpu_runtimes.append(cpu_time)
    cpu_aris.append(adjusted_rand_score(y_true, sk_labels))
    cpu_nmis.append(normalized_mutual_info_score(y_true, sk_labels))

    # =========================================================
    # SAVE RESULTS TO DISK (After each replicate for robustness)
    # =========================================================

    results_dict = {
        "Replicate": list(range(1, i + 2)),
        "CPU_Runtime_sec": cpu_runtimes,
        "CPU_ARI": cpu_aris,
        "CPU_NMI": cpu_nmis
    }

    results_df = pd.DataFrame(results_dict)
    output_filename = "python_gmm_results.csv"
    results_df.to_csv(output_filename, index=False)

# ---------------------------------------------------------
# Final Statistics & Benchmark Results Summary
# ---------------------------------------------------------
cpu_runtime_mean, cpu_runtime_std = np.mean(cpu_runtimes), np.std(cpu_runtimes)
cpu_ari_mean, cpu_ari_std = np.mean(cpu_aris), np.std(cpu_aris)
cpu_nmi_mean, cpu_nmi_std = np.mean(cpu_nmis), np.std(cpu_nmis)

print("\n" + "="*50)
print(f"    GMM Benchmark Summary ({n_replicates} Replicates)    ")
print("-" * 50)
print("Runtime (Seconds):")
print(f"  CPU (sklearn):  {cpu_runtime_mean:.4f} ± {cpu_runtime_std:.4f}")
print("-" * 50)
print("Adjusted Rand Index (ARI):")
print(f"  CPU (sklearn):  {cpu_ari_mean:.6f} ± {cpu_ari_std:.6f}")
print("-" * 50)
print("Normalized Mutual Information (NMI):")
print(f"  CPU (sklearn):  {cpu_nmi_mean:.6f} ± {cpu_nmi_std:.6f}")
print("="*50)
