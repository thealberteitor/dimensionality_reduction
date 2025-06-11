library(smacofx)
library(ggplot2)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

load("data/wine.RData")
data_raw   <- wine[, -1]
labels_raw <- factor(wine[, 1])

# Preprocesado
out    <- preprocess_data(data_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir jitter mínimo
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))

# Cálculo de distancias y lmds
d0 <- dist(X_jitter)
k  <- 15
res_lmds <- lmds(delta = d0, k = k, ndim = 2, tau = 0.5, verbose = 1)
emb <- scale(res_lmds$conf, center = TRUE, scale = FALSE)

# Métricas
d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
sil       <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust     <- compute_trustworthiness(rank0, rank1, k)
cont      <- compute_continuity(rank0, rank1, k)

dir.create("results", showWarnings = FALSE)
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, sil, trust, cont),
    file = "results/metrics_config_lmds_smacofx_wine.txt")

# Plot
df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p  <- ggplot(df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("MDS Local") +
  big_text_theme
print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Local_MDS_lmds_smacofx_Wine_plot.png", plot = p, width = 8, height = 6, dpi = 300)
