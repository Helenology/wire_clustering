# R script: Hierarchical Clustering Benchmark using stats::hclust

library(reticulate)
library(mclust)    # For adjustedRandIndex
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

# NOTE: Hierarchical clustering requires computing a full distance matrix via dist(),
# which has O(n^2) memory complexity. For n_samples=100000, this requires ~80GB RAM.
# If memory is insufficient, reduce n_samples (e.g., to 10000).

runtimes <- c()
aris     <- c()
nmis     <- c()

cat(sprintf("Starting HC Benchmark with %d replicates...\n", n_replicates))
cat(sprintf("Dataset: %d samples, %d features, %d clusters\n", n_samples, n_features, n_clusters))

for (i in 1:n_replicates) {
  cat(sprintf("\n--- Running Replicate %d/%d ---\n", i, n_replicates))
  # Load the pre-generated dataset from disk
  data <- np$load(sprintf("data/replicate_%d.npz", i - 1))
  X      <- data["X"]
  y_true <- data["y"]

  start_time <- proc.time()
  dist_matrix <- dist(X)
  hc_result   <- hclust(dist_matrix, method = "ward.D2")
  r_labels    <- cutree(hc_result, k = n_clusters)
  elapsed <- (proc.time() - start_time)["elapsed"]

  ari <- adjustedRandIndex(y_true, r_labels)
  nmi_val <- NMI(as.integer(y_true), as.integer(r_labels))

  runtimes <- c(runtimes, elapsed)
  aris     <- c(aris, ari)
  nmis     <- c(nmis, nmi_val)

  results_df <- data.frame(Replicate = 1:i, R_Runtime_sec = runtimes, R_ARI = aris, R_NMI = nmis)
  write.csv(results_df, file = "R_hc_results.csv", row.names = FALSE)
}

cat(sprintf("\n%s\n", strrep("=", 50)))
cat(sprintf("    HC R Benchmark Summary (%d Replicates)\n", n_replicates))
cat(sprintf("%s\n", strrep("-", 50)))
cat(sprintf("Runtime: %.4f +/- %.4f\n", mean(runtimes), sd(runtimes)))
cat(sprintf("ARI:     %.6f +/- %.6f\n", mean(aris), sd(aris)))
cat(sprintf("NMI:     %.6f +/- %.6f\n", mean(nmis), sd(nmis)))
cat(sprintf("%s\n", strrep("=", 50)))
