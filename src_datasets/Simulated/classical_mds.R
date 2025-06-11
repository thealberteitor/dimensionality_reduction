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

# MDS Clásico
cmd_emb <- cmdscale(dist(X_j), k = 2)
d1_cmd <- dist(cmd_emb)
d0 <- dist(X_j)
spearman_cmd <- cor(as.vector(d0), as.vector(d1_cmd), method = "spearman")
ari_cmd <- adjustedRandIndex(kmeans(cmd_emb, length(unique(labels)))$cluster, labels)
sil_cmd <- mean(silhouette(kmeans(cmd_emb, length(unique(labels)))$cluster, d1_cmd)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1_cmd <- apply(as.matrix(d1_cmd), 2, rank, ties.method = "average")
trust_cmd <- compute_trustworthiness(rank0, rank1_cmd, k = 7)
cont_cmd <- compute_continuity(rank0, rank1_cmd, k = 7)

# Guardar métricas MDS Clásico
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", 
            spearman_cmd, ari_cmd, sil_cmd, trust_cmd, cont_cmd), 
    file = "results/metrics_mds_clasico_simulated.txt")

plot_df_cmd <- data.frame(Dim1 = cmd_emb[,1], Dim2 = cmd_emb[,2], Label = factor(labels))
p_cmd <- ggplot(plot_df_cmd, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 2, alpha = 0.7) +
  ggtitle("MDS Clásico") +
  big_text_theme

ggsave("plots/MDS_Clasico_simulated.png", plot = p_cmd, dpi = 300, width = 8, height = 6, bg = "white")
