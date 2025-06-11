library(smacofx)
library(mclust)
library(cluster)
library(ggplot2)
library(keras)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

mnist     <- dataset_mnist()
set.seed(123)

n_samples <- 20000
idx        <- sample(seq_len(nrow(mnist$train$x)), n_samples)
data_raw   <- mnist$train$x[idx,,, drop = FALSE]
labels_raw <- mnist$train$y[idx]

# Preprocesado
X_raw <- array_reshape(data_raw, c(n_samples, 28*28)) / 255
out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir jitter mínimo
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

# Cálculo de distancias y lmds
d0 <- dist(X_jitter)
k  <- 15
res_lmds <- lmds(delta = d0, k = k, ndim = 2, tau = 0.5, verbose = 1)
emb <- scale(res_lmds$conf, center = TRUE, scale = FALSE)

# Métricas
d1 <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
sil <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust <- compute_trustworthiness(rank0, rank1, k)
cont <- compute_continuity(rank0, rank1, k)

dir.create("results", showWarnings = FALSE)
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", 
            spearman, ari, sil, trust, cont),
    file = "results/metrics_config_lmds_smacofx_mnist.txt")

# Plot
lmds_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(lmds_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("MDS Local") +
  big_text_theme
print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Local_MDS_lmds_smacofx_MNIST_plot.png", plot = p, dpi = 300, width = 8, height = 6)
