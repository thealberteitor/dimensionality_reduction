library(Rdimtools)
library(ggplot2)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Carga del dataset
load("data/wine.RData")
data_raw   <- wine[, -1]
labels_raw <- factor(wine[, 1])

# Preprocesado
out    <- preprocess_data(data_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir jitter mínimo
set.seed(123)
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

# Sammon Mapping
d0 <- dist(X_j)
sm <- Rdimtools::do.sammon(X_j, ndim = 2)
emb <- sm$Y

# Métricas
d1 <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
sil <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust <- compute_trustworthiness(rank0, rank1, k = 7)
cont  <- compute_continuity(rank0, rank1, k = 7)

dir.create("results", showWarnings = FALSE)
cat(sprintf("Spearman=%.3f\n", spearman), file = "results/metrics_config_sammon_wine.txt")
cat(sprintf("ARI=%.3f\n", ari), file = "results/metrics_config_sammon_wine.txt", append = TRUE)
cat(sprintf("Silhouette=%.3f\n", sil), file = "results/metrics_config_sammon_wine.txt", append = TRUE)
cat(sprintf("Trustworthiness=%.3f\n", trust), file = "results/metrics_config_sammon_wine.txt", append = TRUE)
cat(sprintf("Continuity=%.3f\n", cont), file = "results/metrics_config_sammon_wine.txt", append = TRUE)

# Plot
df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p <- ggplot(df, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("Sammon Mapping") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Sammon_Mapping_Wine_plot.png", plot = p, width = 8, height = 6, dpi = 300)
