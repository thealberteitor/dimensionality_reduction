# mds_no_metrico.R
library(keras)
library(MASS)       
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

N <- 20000
mnist     <- dataset_fashion_mnist()
set.seed(123)
idx       <- sample(seq_len(nrow(mnist$train$x)), N)
data_raw  <- mnist$train$x[idx,,,drop=FALSE]
labels_raw<- mnist$train$y[idx]

# preprocesado previo
data_x <- array_reshape(data_raw, c(N, 28*28)) / 255
out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

#aplicamos nm-mds
d0  <- dist(X_j)
emb <- isoMDS(d0, k = 2)$points

d1           <- dist(emb)
spearman     <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari          <- adjustedRandIndex(kmeans(emb,10)$cluster, labels)
silhouette_m <- mean(silhouette(kmeans(emb,10)$cluster, d1)[, "sil_width"])
rank0        <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1        <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust_nm     <- compute_trustworthiness(rank0, rank1, k = 7)
cont_nm      <- compute_continuity(rank0, rank1, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, silhouette_m, trust_nm, cont_nm),
    file = "results/metrics_config_nonmetric_mnist.txt")

mds_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(mds_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("NM-MDS")
print(p)
ggsave("plots/MDS_NonMetric_fashion_MNIST_plot.png", plot = p, width = 8, height = 6, dpi = 300)
