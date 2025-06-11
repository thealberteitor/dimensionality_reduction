library(Rtsne)
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

# Valores de perplexity a evaluar
perplexities <- c(5, 15, 30, 50)

# Ejecutar y guardar resultados para cada perplexity
for (perp in perplexities) {
  dir.create(paste0("results/perplexity_", perp), showWarnings = FALSE, recursive = TRUE)
  dir.create(paste0("plots/perplexity_", perp), showWarnings = FALSE, recursive = TRUE)

  res_tsne <- Rtsne(X_j, dims = 2, perplexity = perp, max_iter = 1000, check_duplicates = FALSE)
  emb <- res_tsne$Y

  d0 <- dist(X_j)
  d1 <- dist(emb)
  spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
  ari <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
  sil <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
  rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
  rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
  trust <- compute_trustworthiness(rank0, rank1, k = 7)
  cont  <- compute_continuity(rank0, rank1, k = 7)

  # Guardar métricas
  metrics_path <- sprintf("results/perplexity_%d/metrics_tsne.txt", perp)
  cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", 
              spearman, ari, sil, trust, cont), 
      file = metrics_path)

  # Visualización
  plot_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
  p <- ggplot(plot_df, aes(Dim1, Dim2, color = Label)) +
    geom_point(size = 2, alpha = 0.7) +
    ggtitle(paste("t-SNE (Perplexity =", perp, ")")) +
    big_text_theme

  ggsave(sprintf("plots/perplexity_%d/tSNE_plot.png", perp), plot = p, dpi = 300, width = 8, height = 6, bg = "white")
}
