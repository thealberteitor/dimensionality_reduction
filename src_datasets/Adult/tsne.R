# tsne.R

library(keras)
library(Rtsne)
library(mclust)
library(cluster)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Paso 1: Leer y limpiar el dataset Adult Income
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

# Paso 2: Preparar datos y etiquetas
labels_raw <- adult$income
features   <- adult[, setdiff(col_names, "income")]

# convertir factores en dummies y escalar
data_x <- model.matrix(~ . - 1, data = features)
data_x <- scale(data_x)

# Preprocesado previo (elimina outliers, balancea si corresponde)
out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Eliminar duplicados (t-SNE no los admite)
dups <- duplicated(X)
if (any(dups)) {
  X      <- X[!dups, , drop = FALSE]
  labels <- labels[!dups]
}

# aplicamos t-SNE
tsne_res <- Rtsne(X, perplexity = 30, verbose = FALSE, max_iter = 500)
emb      <- tsne_res$Y

# Cálculo de métricas
d0           <- dist(X)
d1           <- dist(emb)
spearman     <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari          <- adjustedRandIndex(kmeans(emb, 2)$cluster, labels)
silhouette_t <- mean(silhouette(kmeans(emb, 2)$cluster, d1)[, "sil_width"])
rank0        <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1        <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust_tn     <- compute_trustworthiness(rank0, rank1, k = 7)
cont_tn      <- compute_continuity(rank0, rank1, k = 7)

# Guardar métricas
cat(
  sprintf(
    "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
    spearman, ari, silhouette_t, trust_tn, cont_tn
  ),
  file = "results/metrics_config_tsne_adult.txt"
)

# Preparar y dibujar el gráfico t-SNE
tsne_df <- data.frame(
  Dim1  = emb[,1],
  Dim2  = emb[,2],
  Label = factor(labels)
)

p <- ggplot(tsne_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("t-SNE")

print(p)
ggsave("plots/tSNE_Adult_plot.png", plot = p, width = 8, height = 6, dpi = 300)
