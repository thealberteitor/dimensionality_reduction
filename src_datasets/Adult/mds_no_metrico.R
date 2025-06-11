# mds_nonmetric_adult.R

# Cargar librerías
library(MASS)       # isoMDS()
library(mclust)     # adjustedRandIndex()
library(cluster)    # silhouette()
library(ggplot2)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

# Leer y limpiar el dataset Adult Census Income
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

# Preparar datos y etiquetas
labels_raw <- adult$income
features   <- adult[, setdiff(names(adult), "income")]

# Convertir factores en dummies y escalar
X_raw  <- model.matrix(~ . - 1, data = features)
data_x <- scale(X_raw)

# Preprocesado: eliminar outliers, balanceo si corresponde
out    <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir ruido mínimo para romper empates en distancias
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))


d0  <- dist(X_j)
res <- isoMDS(d0, k = 2)
emb <- res$points

d1 <- dist(emb)


# Cálculo de métricas de preservación de estructura
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari      <- adjustedRandIndex(kmeans(emb, 2)$cluster, labels)
sil      <- mean(silhouette(kmeans(emb, 2)$cluster, d1)[, "sil_width"])
rank0    <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1    <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust_nm <- compute_trustworthiness(rank0, rank1, k = 7)
cont_nm  <- compute_continuity(rank0, rank1, k = 7)



cat(
  sprintf(
    "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
    spearman, ari, sil, trust_nm, cont_nm
  ),
  file = "results/metrics_config_nonmetric_adult.txt"
)


mds_df <- data.frame(
  Dim1  = emb[, 1],
  Dim2  = emb[, 2],
  Label = factor(labels)
)
p <- ggplot(mds_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("NM-MDS") +
  big_text_theme

print(p)
ggsave(
  filename = "plots/MDS_NonMetric_adult_plot.png",
  plot     = p,
  width    = 8,
  height   = 6,
  dpi      = 300
)
