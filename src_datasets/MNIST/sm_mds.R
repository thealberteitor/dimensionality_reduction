library(Rdimtools)
library(mclust)
library(cluster)
library(ggplot2)
library(keras)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Carga del dataset
N <- 20000
mnist     <- dataset_mnist()
set.seed(123)
idx        <- sample(seq_len(nrow(mnist$train$x)), N)
data_raw   <- mnist$train$x[idx,,,drop=FALSE]
labels_raw <- mnist$train$y[idx]

data_x <- array_reshape(data_raw, c(N, 28*28)) / 255

# Preprocesado
out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir jitter mínimo
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

# Sammon Mapping
d0 <- dist(X_j)
sm <- Rdimtools::do.sammon(X_j, ndim = 2)
emb <- sm$Y

# Métricas
d1 <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari <- adjustedRandIndex(kmeans(emb, 10)$cluster, labels)
sil <- mean(silhouette(kmeans(emb, 10)$cluster, d1)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust <- compute_trustworthiness(rank0, rank1, k = 7)
cont  <- compute_continuity(rank0, rank1, k = 7)

dir.create("results", showWarnings = FALSE)
cat(sprintf("Spearman=%.3f\n", spearman), file = "results/metrics_config_sammon_mnist.txt")
cat(sprintf("ARI=%.3f\n", ari), file = "results/metrics_config_sammon_mnist.txt", append = TRUE)
cat(sprintf("Silhouette=%.3f\n", sil), file = "results/metrics_config_sammon_mnist.txt", append = TRUE)
cat(sprintf("Trustworthiness=%.3f\n", trust), file = "results/metrics_config_sammon_mnist.txt", append = TRUE)
cat(sprintf("Continuity=%.3f\n", cont), file = "results/metrics_config_sammon_mnist.txt", append = TRUE)

# Plot
sam_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(sam_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("Sammon Mapping") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Sammon_Mapping_mnist_plot.png", plot = p, width = 8, height = 6, dpi = 300)
