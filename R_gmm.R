# R script: GMM Benchmark using mclust::Mclust
#setwd("Documents/wire")
library(reticulate)
library(mclust)
library(aricode)   # For NMI calculation: install.packages("aricode")

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

runtimes <- c()
aris     <- c()
nmis     <- c()

cat(sprintf("Starting GMM Benchmark with %d replicates...\n", n_replicates))
cat(sprintf("Dataset: %d samples, %d features, %d clusters\n", n_samples, n_features, n_clusters))

for (i in 1:n_replicates) {
  cat(sprintf("\n--- Running Replicate %d/%d ---\n", i, n_replicates))
  
  # Load the pre-generated dataset from disk
  data <- np$load(sprintf("data/replicate_%d.npz", i - 1))
  X      <- data["X"]
  y_true <- as.integer(data["y"])
  
  set.seed(i)
  start_time <- proc.time()
  
  # Optimization for mclust:
  # 1. modelNames = "EII" (Spherical, equal volume) to match your isotropic blobs
  # 2. initialization = list(subset = ...) to bypass the N^2 hierarchical clustering
  gmm_result <- Mclust(
    X, 
    G = n_clusters, 
    modelNames = "EII", # Force spherical model to improve stability and speed
    initialization = list(subset = sample(1:nrow(X), 500)) # Small subset for init
  )
  
  elapsed <- (proc.time() - start_time)["elapsed"]
  
  # Predicted labels
  r_labels <- as.integer(gmm_result$classification)
  
  # Calculate ARI and NMI
  ari_val <- adjustedRandIndex(y_true, r_labels)
  nmi_val <- NMI(y_true, r_labels)
  
  runtimes <- c(runtimes, elapsed)
  aris     <- c(aris, ari_val)
  nmis     <- c(nmis, nmi_val)
  
  # Save incremental results
  results_df <- data.frame(Replicate = 1:i, R_Runtime_sec = runtimes, R_ARI = aris, R_NMI = nmis)
  write.csv(results_df, file = "R_gmm_results.csv", row.names = FALSE)
}

cat(sprintf("\nSummary: Runtime: %.4f +/- %.4f, ARI: %.6f\n", mean(runtimes), sd(runtimes), mean(aris)))
cat(sprintf("\n%s\n", strrep("=", 50)))
cat(sprintf("    GMM R Benchmark Summary (%d Replicates)\n", n_replicates))
cat(sprintf("%s\n", strrep("-", 50)))
cat(sprintf("Runtime: %.4f +/- %.4f\n", mean(runtimes), sd(runtimes)))
cat(sprintf("ARI:     %.6f +/- %.6f\n", mean(aris), sd(aris)))
cat(sprintf("NMI:     %.6f +/- %.6f\n", mean(nmis), sd(nmis)))
cat(sprintf("%s\n", strrep("=", 50)))
