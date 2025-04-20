library(smacof) 
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

# Matriz de distancias y pesos locales (k-NN)
d0    <- dist(X)
mat   <- as.matrix(d0)
n     <- nrow(mat)
k_nn  <- 7
nn    <- apply(mat, 1, function(x) order(x)[2:(k_nn+1)])
W     <- matrix(0, n, n)
for(i in seq_len(n)) {
  W[i,   nn[,i]] <- 1
  W[nn[,i], i]   <- 1
}

# Local MDS
res_lmds <- smacofSym(d0, weightmat = W, ndim = 2)
emb      <- res_lmds$conf


d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
sil_obj   <- silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)
sil       <- mean(sil_obj[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust     <- compute_trustworthiness(rank0, rank1, k_nn)
cont      <- compute_continuity(rank0, rank1, k_nn)

cat(sprintf("Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
            spearman, ari, sil, trust, cont),
    file = "results/metrics_config_local_mds_wine.txt")

df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Class = labels)
p  <- ggplot(df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("L‑MDS") +
  big_text_theme
print(p)
ggsave("plots/Local_MDS_Wine_plot.png", plot = p, width = 8, height = 6, dpi = 300)
