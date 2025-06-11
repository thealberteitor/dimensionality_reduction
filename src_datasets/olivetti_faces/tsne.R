library(RnavGraphImageData)
library(snedata)
library(Rtsne)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

df_raw    <- olivetti_faces()
X_raw     <- df_raw[, grep("^px", names(df_raw))]
labels_raw<- factor(df_raw$Label)

# Preprocess
out       <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X         <- out$data
labels    <- out$labels


set.seed(123)
tsne_out  <- Rtsne(X, perplexity = 30, verbose = FALSE, max_iter = 1000)
emb       <- tsne_out$Y

# metrics
d0        <- dist(X)
d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
sil       <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust     <- compute_trustworthiness(rank0, rank1, k = 7)
cont      <- compute_continuity(rank0, rank1, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, sil, trust, cont),
    file = "results/metrics_tsne_olivetti.txt")

df_plot   <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p         <- ggplot(df_plot, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 1.5) +
  ggtitle("t-SNE")
print(p)
ggsave("plots/tsne_olivetti_plot.png", plot = p, width = 8, height = 6, dpi = 300)
