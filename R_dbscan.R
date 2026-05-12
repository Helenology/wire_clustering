# R script: DBSCAN Benchmark using dbscan::dbscan

library(reticulate)
library(mclust)    # For adjustedRandIndex
library(dbscan)    # For dbscan()
library(aricode)   # For NMI

# Load numpy for reading pre-generated .npz files
np <- import("numpy")

# ---------------------------------------------------------
# Load Experiment Parameters from config.json
# ---------------------------------------------------------
config       <- jsonlite::fromJSON("data/config.json")
n_samples    <- as.integer(config$n_samples)
n_features   <- as.integer(config$n_features)
n_clusters   <- as.integer(config$n_clusters)
n_replicates <- as.integer(config$n_replicates)

# DBSCAN hyperparameters
# For 500-dimensional Gaussian blobs with cluster_std in [2, 10],
# within-cluster Euclidean distances scale as ~ std * sqrt(n_features).
# The tightest cluster (std=2) has mean pairwise distance ~ 2*sqrt(500) ~ 63.
# We set eps to capture dense neighborhoods within the tightest clusters.
dbscan_eps  <- 75.0
dbscan_minPts <- 10L

runtimes <- c()
aris     <- c()
nmis     <- c()

cat(sprintf("Starting DBSCAN Benchmark with %d replicates...\n", n_replicates))
cat(sprintf("Dataset: %d samples, %d features, %d clusters\n", n_samples, n_features, n_clusters))
cat(sprintf("DBSCAN params: eps=%.1f, minPts=%d\n", dbscan_eps, dbscan_minPts))

for (i in 1:n_replicates) {
  cat(sprintf("\n--- Running Replicate %d/%d ---\n", i, n_replicates))
  # Load the pre-generated dataset from disk
  data <- np$load(sprintf("data/replicate_%d.npz", i - 1))
  X      <- data["X"]
  y_true <- data["y"]

  start_time <- proc.time()
  dbscan_result <- dbscan(X, eps = dbscan_eps, minPts = dbscan_minPts)
  elapsed <- (proc.time() - start_time)["elapsed"]

  r_labels <- dbscan_result$cluster
  ari <- adjustedRandIndex(y_true, r_labels)
  nmi_val <- NMI(as.integer(y_true), as.integer(r_labels))

  runtimes <- c(runtimes, elapsed)
  aris     <- c(aris, ari)
  nmis     <- c(nmis, nmi_val)

  results_df <- data.frame(Replicate = 1:i, R_Runtime_sec = runtimes, R_ARI = aris, R_NMI = nmis)
  write.csv(results_df, file = "R_dbscan_results.csv", row.names = FALSE)
}

cat(sprintf("\n%s\n", strrep("=", 50)))
cat(sprintf("    DBSCAN R Benchmark Summary (%d Replicates)\n", n_replicates))
cat(sprintf("%s\n", strrep("-", 50)))
cat(sprintf("Runtime: %.4f +/- %.4f\n", mean(runtimes), sd(runtimes)))
cat(sprintf("ARI:     %.6f +/- %.6f\n", mean(aris), sd(aris)))
cat(sprintf("NMI:     %.6f +/- %.6f\n", mean(nmis), sd(nmis)))
cat(sprintf("%s\n", strrep("=", 50)))
