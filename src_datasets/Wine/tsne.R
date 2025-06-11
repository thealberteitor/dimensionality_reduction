library(Rtsne)
library(ggplot2)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

load("data/wine.RData")
data_raw   <- wine[, -1]
labels_raw <- factor(wine[, 1])

#  Preprocesado
out    <- preprocess_data(data_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# t‑SNE
set.seed(123)
tsne_res <- Rtsne(X, perplexity = 30, verbose = FALSE, max_iter =1000)
emb      <- tsne_res$Y

d0        <- dist(X)
d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
silhouette_t <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust     <- compute_trustworthiness(rank0, rank1, k = 7)
cont      <- compute_continuity(rank0, rank1, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, silhouette_t, trust, cont),
    file = "results/metrics_config_tsne_wine.txt")

# 6) Plot
df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p  <- ggplot(df, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("t‑SNE")
print(p)
ggsave("plots/tSNE_Wine_plot.png", plot = p, width = 8, height = 6, dpi = 300)
