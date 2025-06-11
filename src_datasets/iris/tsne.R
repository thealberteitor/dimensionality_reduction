# tSNE.R
library(Rtsne)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")


data(iris)
out <- preprocess_data(iris[,1:4], iris$Species, z_threshold = 3)
iris_data <- out$data; labels <- out$labels

# jitter mínimo para evitar distancias zero
set.seed(123)
eps <- 1e-8
iris_j <- iris_data + matrix(rnorm(nrow(iris_data)*ncol(iris_data), sd = eps),
                             nrow = nrow(iris_data))

# 1) t-SNE (Barnes-Hut)
tsne_res <- Rtsne(iris_j, perplexity = 30, verbose = FALSE, max_iter = 1000)
emb <- tsne_res$Y

# 2) Métricas
d0 <- dist(iris_j)
d1 <- dist(emb)
cat(sprintf("Spearman=%.3f\n", cor(as.vector(d0), as.vector(d1), method = "spearman")),
    file = "results/metrics_config_tsne.txt")
cat(sprintf("ARI=%.3f\n", adjustedRandIndex(kmeans(emb,3)$cluster, labels)),
    file = "results/metrics_config_tsne.txt", append = TRUE)
cat(sprintf("Silhouette=%.3f\n", mean(silhouette(kmeans(emb,3)$cluster, d1)[,"sil_width"])),
    file = "results/metrics_config_tsne.txt", append = TRUE)

rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
k <- 7
cat(sprintf("Trustworthiness=%.3f\n", compute_trustworthiness(rank0, rank1, k)),
    file = "results/metrics_config_tsne.txt", append = TRUE)
cat(sprintf("Continuity=%.3f\n",     compute_continuity(rank0, rank1, k)),
    file = "results/metrics_config_tsne.txt", append = TRUE)

# 3) Plot
tsne_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Species = labels)
p <- ggplot(tsne_df, aes(Dim1, Dim2, color = Species)) +
  geom_point(size = 2) +
  ggtitle("t‑SNE")

print(p)
ggsave("plots/tSNE_iris.png", plot = p, dpi = 300, width = 8, height = 6)
