# C-MDS.R
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")


data(iris)
out <- preprocess_data(iris[,1:4], iris$Species, z_threshold = 3)
iris_data <- out$data; labels <- out$labels

# 1) Cálculo de C-MDS
d0 <- dist(iris_data)
mds2 <- cmdscale(d0, k = 2)
mds_df <- data.frame(Dim1 = mds2[,1], Dim2 = mds2[,2], Species = labels)
d1 <- dist(mds2)

# Spearman
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
cat(sprintf("spearman=%.3f\n", spearman), file = "results/metrics_config_cmds.txt")

# ARI
ari <- adjustedRandIndex(kmeans(mds2, centers = 3)$cluster, labels)
cat(sprintf("ari=%.3f\n", ari), file = "results/metrics_config_cmds.txt", append = TRUE)

# Silhouette
sil <- silhouette(kmeans(mds2, centers = 3)$cluster, d1)
cat(sprintf("silhouette=%.3f\n", mean(sil[,"sil_width"])), "\n",
    file = "results/metrics_config_cmds.txt", append = TRUE)

# Trustworthiness & Continuity
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
k <- 7
cat(sprintf("trustworthiness=%.3f\n", compute_trustworthiness(rank0, rank1, k)),
    file = "results/metrics_config_cmds.txt", append = TRUE)
cat(sprintf("continuity=%.3f\n",     compute_continuity(rank0, rank1, k)),
    file = "results/metrics_config_cmds.txt", append = TRUE)


p <- ggplot(mds_df, aes(Dim1, Dim2, color = Species)) +
  geom_point(size = 2) +
  ggtitle("C‑MDS")

print(p)
ggsave("plots/C-MDS_plot.png", plot = p, dpi = 300, width = 8, height = 6)
