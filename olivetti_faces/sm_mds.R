library(RnavGraphImageData)
library(snedata)
library(MASS)     
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Preprocessed X and labels
df    <- olivetti_faces()
X_raw <- df[, grep("^px", names(df))]
labels_raw <- factor(df$Label)
i_out <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X     <- i_out$data
labels<- i_out$labels

set.seed(123)
epsilon <- 1e-8
X_j     <- X + matrix(rnorm(nrow(X)*ncol(X), sd = epsilon), nrow = nrow(X))

# 3) Sammon mapping
sm      <- sammon(dist(X_j), k = 2)
emb_s   <- sm$points

# 4) Metrics
d1_s       <- dist(emb_s)
spearman_s <- cor(as.vector(d0), as.vector(d1_s), method = "spearman")
ari_s      <- adjustedRandIndex(kmeans(emb_s, length(unique(labels)))$cluster, labels)
sil_s      <- mean(silhouette(kmeans(emb_s, length(unique(labels)))$cluster, d1_s)[, "sil_width"])
rank1_s    <- apply(as.matrix(d1_s), 2, rank, ties.method = "average")
trust_s    <- compute_trustworthiness(rank0, rank1_s, k = 7)
cont_s     <- compute_continuity(rank0, rank1_s, k = 7)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman_s, ari_s, sil_s, trust_s, cont_s),
    file = "results/metrics_sammon_olivetti.txt")

df_s <- data.frame(Dim1 = emb_s[,1], Dim2 = emb_s[,2], Class = labels)
p_s  <- ggplot(df_s, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 1) +
  ggtitle("SM-MDS")
print(p_s)
ggsave("plots/Sammon_Olivetti.png", plot = p_s, width = 8, height = 6, dpi = 300)
