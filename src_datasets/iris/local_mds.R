#-------------------------------
# Paso 1: Carga y preprocesado
#-------------------------------
library(smacofx)
library(mclust)
library(cluster)
library(ggplot2)

source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

out    <- preprocess_data(iris[,1:4], iris$Species, z_threshold = 3)
X      <- out$data
labels <- out$labels


set.seed(123)
eps      <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps),
                       nrow = nrow(X))

d0 <- dist(X_jitter)
n  <- nrow(as.matrix(d0))
k  <- 15

# lmds construye internamente W según k y tau
res_lmds <- lmds(
  delta   = d0,
  k       = k,
  ndim    = 2,
  tau     = 0.5,
  verbose = 1
)
emb <- res_lmds$conf

#----------------------------------------
# Paso 3: Métricas, guardar y plot
#----------------------------------------
# Centrar para visualización
emb_cent <- scale(emb, center = TRUE, scale = FALSE)

# Métricas de calidad
d1         <- dist(emb_cent)
spearman   <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari        <- adjustedRandIndex(kmeans(emb_cent, 3)$cluster, labels)
sil        <- mean(silhouette(kmeans(emb_cent, 3)$cluster, d1)[, "sil_width"])
rank0      <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1      <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust      <- compute_trustworthiness(rank0, rank1, k)
continuity <- compute_continuity      (rank0, rank1, k)

dir.create("results", showWarnings = FALSE)
cat(sprintf(
  "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
  spearman, ari, sil, trust, continuity
), file = "results/metrics_config_lmds_smacofx_iris.txt")

# Plot
mds_df <- data.frame(Dim1 = emb_cent[,1],
                     Dim2 = emb_cent[,2],
                     Species = labels)

p <- ggplot(mds_df, aes(Dim1, Dim2, color = Species)) +
  geom_point(size = 2) +
  ggtitle("MDS Local") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Local_MDS_lmds_smacofx_Iris_plot.png",
       plot = p, dpi = 300, width = 8, height = 6)
