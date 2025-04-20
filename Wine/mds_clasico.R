library(ggplot2)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")


load("data/wine.RData")         
data_raw <- wine[, -1]         
labels_raw <- factor(wine[, 1]) 

# Preprocesado
out    <- preprocess_data(data_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Embedding MDS clásico
d0  <- dist(X)
emb <- cmdscale(d0, k = 2)


d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
sil       <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust     <- compute_trustworthiness(rank0, rank1, k = 7)
cont      <- compute_continuity(rank0, rank1, k = 7)

# métricas
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, sil, trust, cont),
    file = "results/metrics_config_wine.txt")


df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p  <- ggplot(df, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("C-MDS")
print(p)
ggsave("plots/C-MDS_Wine_plot.png", plot = p, width = 8, height = 6, dpi = 300)
