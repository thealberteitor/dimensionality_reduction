library(MASS)
library(cluster)
library(mclust)
library(ggplot2)
source("../metrics.R")
source("../plot_theme.R")

# Cargar datos simulados
sim_data <- readRDS("simulated_data_list.rds")
X <- sim_data$data_scaled
labels <- sim_data$labels

# Añadir jitter
set.seed(123)
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))

# Sammon Mapping
res_sammon <- sammon(dist(X_j), k = 2)
sam_emb <- res_sammon$points
d0 <- dist(X_j)
d1_sam <- dist(sam_emb)
spearman_sam <- cor(as.vector(d0), as.vector(d1_sam), method = "spearman")
ari_sam <- adjustedRandIndex(kmeans(sam_emb, length(unique(labels)))$cluster, labels)
sil_sam <- mean(silhouette(kmeans(sam_emb, length(unique(labels)))$cluster, d1_sam)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1_sam <- apply(as.matrix(d1_sam), 2, rank, ties.method = "average")
trust_sam <- compute_trustworthiness(rank0, rank1_sam, k = 7)
cont_sam <- compute_continuity(rank0, rank1_sam, k = 7)

# Guardar métricas Sammon
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", 
            spearman_sam, ari_sam, sil_sam, trust_sam, cont_sam), 
    file = "results/metrics_sammon_simulated.txt")

plot_df_sam <- data.frame(Dim1 = sam_emb[,1], Dim2 = sam_emb[,2], Label = factor(labels))
p_sam <- ggplot(plot_df_sam, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 2, alpha = 0.7) +
  ggtitle("Sammon Mapping") +
  big_text_theme

ggsave("plots/Sammon_Mapping_simulated.png", plot = p_sam, dpi = 300, width = 8, height = 6, bg = "white")
