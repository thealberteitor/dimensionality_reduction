library(Rdimtools)
library(mclust)
library(cluster)
library(ggplot2)
library(keras)
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
features   <- adult[, setdiff(names(adult), "income")]
X_raw      <- model.matrix(~ . - 1, data = features)

out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# Añadir jitter mínimo
set.seed(123)
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps), nrow = nrow(X))

# Paso 3: Sammon Mapping
d0 <- dist(X_j)
sm <- Rdimtools::do.sammon(X_j, ndim = 2)
emb <- sm$Y

# Paso 4: Métricas
d1 <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari <- adjustedRandIndex(kmeans(emb, length(unique(labels)))$cluster, labels)
sil <- mean(silhouette(kmeans(emb, length(unique(labels)))$cluster, d1)[, "sil_width"])
rank0 <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
trust <- compute_trustworthiness(rank0, rank1, k = 7)
cont  <- compute_continuity(rank0, rank1, k = 7)

dir.create("results", showWarnings = FALSE)
cat(sprintf("Spearman=%.3f\n", spearman), file = "results/metrics_config_sammon_adult.txt")
cat(sprintf("ARI=%.3f\n", ari), file = "results/metrics_config_sammon_adult.txt", append = TRUE)
cat(sprintf("Silhouette=%.3f\n", sil), file = "results/metrics_config_sammon_adult.txt", append = TRUE)
cat(sprintf("Trustworthiness=%.3f\n", trust), file = "results/metrics_config_sammon_adult.txt", append = TRUE)
cat(sprintf("Continuity=%.3f\n", cont), file = "results/metrics_config_sammon_adult.txt", append = TRUE)

# Paso 5: Gráfico
sam_df <- data.frame(Dim1 = emb[,1], Dim2 = emb[,2], Label = factor(labels))
p <- ggplot(sam_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("Sammon Mapping") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Sammon_Mapping_adult_plot.png", plot = p, dpi = 300, width = 8, height = 6)
