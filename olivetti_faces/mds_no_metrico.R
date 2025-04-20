# mds_nonmetric_olivetti.R
library(RnavGraphImageData)
library(snedata)
library(MASS)        # for isoMDS()
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Preprocess
orf   <- olivetti_faces()
X_raw <- orf[, grep("^px", names(orf))]
labels_raw <- factor(orf$Label)
i_out <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X     <- i_out$data
labels<- i_out$labels

# Add jitter to avoid zero distances
set.seed(123)
epsilon <- 1e-8
X_j     <- X + matrix(rnorm(nrow(X)*ncol(X), sd = epsilon), nrow = nrow(X))

d0     <- dist(X)

# Non-metric MDS embedding
iso    <- isoMDS(dist(X_j), k = 2)
emb_n  <- iso$points

# Metrics
d1_n       <- dist(emb_n)
spearman_n <- cor(as.vector(d0), as.vector(d1_n), method = "spearman")
ari_n      <- adjustedRandIndex(kmeans(emb_n, length(unique(labels)))$cluster, labels)
sil_n      <- mean(silhouette(kmeans(emb_n, length(unique(labels)))$cluster, d1_n)[, "sil_width"])
rank1_n    <- apply(as.matrix(d1_n), 2, rank, ties.method = "average")
trust_n    <- compute_trustworthiness(rank0, rank1_n, k = 7)
cont_n     <- compute_continuity(rank0, rank1_n, k = 7)


cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman_n, ari_n, sil_n, trust_n, cont_n),
    file = "results/metrics_nonmetric_olivetti.txt")



df_n <- data.frame(Dim1 = emb_n[,1], Dim2 = emb_n[,2], Class = labels)
p_n  <- ggplot(df_n, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 1) +
  ggtitle("NM-MDS")
print(p_n)
ggsave("plots/NM-MDS_Olivetti.png", plot = p_n, width = 8, height = 6, dpi = 300)
