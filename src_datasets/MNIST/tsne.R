# tsne.R
library(keras)
library(Rtsne)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

N <- 20000
mnist     <- dataset_mnist()
set.seed(123)
idx       <- sample(seq_len(nrow(mnist$train$x)), N)
data_raw  <- mnist$train$x[idx,,,drop=FALSE]
labels_raw<- mnist$train$y[idx]

# Preprocesado previo
data_x <- array_reshape(data_raw, c(N, 28*28)) / 255
out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# aplicamos tsne
tsne_res <- Rtsne(X, perplexity = 30, verbose = FALSE, max_iter = 1000)
emb      <- tsne_res$Y

d0           <- dist(X)
d1           <- dist(emb)
spearman     <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari          <- adjustedRandIndex(kmeans(emb,10)$cluster, labels)
silhouette_t <- mean(silhouette(kmeans(emb,10)$cluster, d1)[, "sil_width"])
rank0        <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1        <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust_tn     <- compute_trustworthiness(rank0, rank1, k = 7)
cont_tn      <- compute_continuity(rank0, rank1, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, silhouette_t, trust_tn, cont_tn),
    file = "results/metrics_config_tsne_mnist.txt")

tsne_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(tsne_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("t‑SNE")
print(p)
ggsave("plots/tSNE_MNIST_plot.png", plot = p, width = 8, height = 6, dpi = 300)
