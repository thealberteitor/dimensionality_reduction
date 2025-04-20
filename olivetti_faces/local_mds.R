library(RnavGraphImageData)
library(MASS) 
library(smacof)
library(mclust)
library(cluster)
library(ggplot2)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")


df_raw <- olivetti_faces()
X_raw   <- df_raw[, grep("^px", names(df_raw))]
labels_raw <- factor(df_raw$Label)
out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# matriz de distancias y pesos locales (k-NN)
d0     <- dist(X)
mat_d0 <- as.matrix(d0)
n      <- nrow(mat_d0)
k      <- 7

# encontrar k vecinos más cercanos para cada punto
nn_idx <- apply(mat_d0, 1, function(x) order(x)[2:(k+1)])

W <- matrix(0, n, n)
for(i in seq_len(n)){
  W[i, nn_idx[,i]] <- 1
  W[nn_idx[,i], i] <- 1
}

# Local MDS
res_lmds <- smacofSym(d0, weightmat = W, ndim = 2)
emb      <- res_lmds$conf


#Métricas
d1       <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari      <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
sil      <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0    <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1    <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust    <- compute_trustworthiness(rank0, rank1, k)
cont     <- compute_continuity(rank0, rank1, k)
cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n", spearman, ari, sil, trust, cont),
    file = "results/metrics_config_local_mds_olivetti.txt")


lmds_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p <- ggplot(lmds_df, aes(Dim1, Dim2, color = Class)) +
  geom_point(size = 1) +
  ggtitle("L-MDS") +
  big_text_theme
print(p)
ggsave("plots/Local_MDS_Olivetti_plot.png", plot = p, dpi = 300, width = 8, height = 6)
