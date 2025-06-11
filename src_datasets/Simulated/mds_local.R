library(smacofx)
library(mclust)
library(cluster)
library(ggplot2)
source("../metrics.R")
source("../plot_theme.R")

# Cargar datos simulados
sim_data <- readRDS("simulated_data_list.rds")
X <- sim_data$data_scaled
labels <- sim_data$labels

# Añadir jitter
set.seed(123)
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))

# Distancia y matriz de vecinos
d0 <- dist(X_j)
mat_d0 <- as.matrix(d0)
n <- nrow(mat_d0)

# Valores de k y tau a explorar
k_vals <- c(5, 15, 30)
tau_vals <- c(0.3, 0.5, 0.7)

# Iterar sobre combinaciones de k y tau
for (k in k_vals) {
  nn_idx <- apply(mat_d0, 1, function(x) order(x)[2:(k+1)])
  W <- matrix(0, n, n)
  for (i in seq_len(n)) {
    W[i, nn_idx[,i]] <- 1
    W[nn_idx[,i], i] <- 1
  }

  for (tau in tau_vals) {
    dir_path <- sprintf("results/local_mds_k%d_tau%.1f", k, tau)
    plot_path <- sprintf("plots/local_mds_k%d_tau%.1f", k, tau)
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    dir.create(plot_path, recursive = TRUE, showWarnings = FALSE)

    res_lmds <- lmds(delta = d0, weightmat = W, k = k, ndim = 2, tau = tau, verbose = 2)
    emb <- res_lmds$conf

    d1 <- dist(emb)
    spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
    ari <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
    sil <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
    rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
    rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
    trust <- compute_trustworthiness(rank0, rank1, k = 7)
    cont  <- compute_continuity(rank0, rank1, k = 7)

    cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", 
                spearman, ari, sil, trust, cont), 
        file = file.path(dir_path, "metrics_lmds.txt"))

    plot_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
    p <- ggplot(plot_df, aes(Dim1, Dim2, color = Label)) +
      geom_point(size = 2, alpha = 0.7) +
      ggtitle(sprintf("Local MDS (k=%d, tau=%.1f)", k, tau)) +
      big_text_theme

    ggsave(file.path(plot_path, "Local_MDS_plot.png"), plot = p, dpi = 300, width = 8, height = 6, bg = "white")
  }
}
