library(RnavGraphImageData)
library(snedata)        
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

df_raw <- olivetti_faces()
X_raw  <- df_raw[, grep("^px", names(df_raw))]
labels_raw <- factor(df_raw$Label)

# Preprocess
i_out  <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- i_out$data
labels <- i_out$labels

# Classical MDS embedding
d0    <- dist(X)
emb_c <- cmdscale(d0, k = 2)

# Metrics
d1_c         <- dist(emb_c)
spearman_c   <- cor(as.vector(d0), as.vector(d1_c), method = "spearman")
ari_c        <- adjustedRandIndex(kmeans(emb_c, length(unique(labels)))$cluster, labels)
sil_c        <- mean(silhouette(kmeans(emb_c, length(unique(labels)))$cluster, d1_c)[, "sil_width"])
rank0        <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1_c      <- apply(as.matrix(d1_c), 2, rank, ties.method = "average")
trust_c      <- compute_trustworthiness(rank0, rank1_c, k = 7)
cont_c       <- compute_continuity(rank0, rank1_c, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman_c, ari_c, sil_c, trust_c, cont_c),
    file = "results/metrics_cmds_olivetti.txt")

df_c <- data.frame(Dim1 = emb_c[,1], Dim2 = emb_c[,2], Class = labels)
p_c  <- ggplot(df_c, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 1) +
  ggtitle("C-MDS")
print(p_c)
ggsave("plots/C-MDS_Olivetti.png", plot = p_c, width = 8, height = 6, dpi = 300)
