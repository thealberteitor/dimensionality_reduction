library(smacofx)
library(mclust)
library(cluster)
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

labels_raw <- adult$income
features   <- adult[, setdiff(names(adult), "income")]
X_raw      <- model.matrix(~ . - 1, data = features)
data_x     <- scale(X_raw)
out        <- preprocess_data(data_x, labels_raw, z_threshold = 3)
X          <- out$data
labels     <- out$labels

# Añadir jitter pequeño para evitar distancias cero
set.seed(123)
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

# Cálculo de distancias y LMDS
d0 <- dist(X_jitter)
k  <- 15
res_lmds <- lmds(delta = d0, k = k, ndim = 2, tau = 0.5, verbose = 1)
emb <- scale(res_lmds$conf, center = TRUE, scale = FALSE)

# Métricas
d1 <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari <- adjustedRandIndex(kmeans(emb, centers = length(unique(labels)))$cluster, labels)
sil <- mean(silhouette(kmeans(emb, centers = length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust <- compute_trustworthiness(rank0, rank1, k)
continuity <- compute_continuity(rank0, rank1, k)

dir.create("results", showWarnings = FALSE)
cat(sprintf(
  "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
  spearman, ari, sil, trust, continuity
), file = "results/metrics_config_lmds_smacofx_adult.txt")

# Plot
lmds_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(lmds_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 2) +
  ggtitle("MDS Local") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Local_MDS_lmds_smacofx_adult_plot.png", plot = p, dpi = 300, width = 8, height = 6)
