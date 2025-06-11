# mds_clasico.R

library(keras)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Leer y limpiar el dataset Adult Income
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names <- c(
  "age","workclass","fnlwgt","education","education_num","marital_status",
  "occupation","relationship","race","sex","capital_gain","capital_loss",
  "hours_per_week","native_country","income"
)
adult <- read.csv(
  url,
  header           = FALSE,
  col.names        = col_names,
  na.strings       = "?",
  stringsAsFactors = TRUE
)
adult <- na.omit(adult)

labels_raw <- adult$income
features   <- adult[, setdiff(names(adult), "income")]
X_raw      <- model.matrix(~ . - 1, data = features)
data_x     <- X_raw

out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Distancias y Classical MDS
d0  <- dist(X)
emb <- cmdscale(d0, k = 2)

# Métricas
d1        <- dist(emb)
spearman  <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari       <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
sil       <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0     <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1     <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust_mds <- compute_trustworthiness(rank0, rank1, k = 7)
cont_mds  <- compute_continuity(rank0, rank1, k = 7)

cat(
  sprintf(
    "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
    spearman, ari, sil, trust_mds, cont_mds
  ),
  file = "results/metrics_config_mds_adult.txt"
)

# Plot
mds_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(mds_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("C-MDS") +
  big_text_theme
  
print(p)
ggsave("plots/C-MDS_adult_plot.png", plot = p, width = 8, height = 6, dpi = 300)
