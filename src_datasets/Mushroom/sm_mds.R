#!/usr/bin/env Rscript

# sm_mds.R (Sammon Mapping with Rdimtools)
library(Rdimtools)   # Cambio: usar do.sammon
library(mclust)
library(cluster)
library(ggplot2)
source("../metrics.R")
source("../preprocess.R")
source("../plot_theme.R")

#-------------------------------
# Paso 1: Carga y preprocesado
#-------------------------------
col_names <- c(
  "class","cap_shape","cap_surface","cap_color","bruises","odor",
  "gill_attachment","gill_spacing","gill_size","gill_color",
  "stalk_shape","stalk_root",
  "stalk_surface_above_ring","stalk_surface_below_ring",
  "stalk_color_above_ring","stalk_color_below_ring",
  "veil_type","veil_color","ring_number","ring_type",
  "spore_print_color","population","habitat"
)
mushroom <- read.csv("data/mushroom.csv",
                     header = FALSE,
                     col.names = col_names,
                     stringsAsFactors = TRUE)

# Eliminar columnas constantes
constant_cols <- sapply(mushroom, function(col)
  is.factor(col) && nlevels(col) < 2)
mushroom <- mushroom[, !constant_cols]

labels_raw <- mushroom$class
features   <- mushroom[, setdiff(names(mushroom), "class")]
X_raw      <- model.matrix(~ . - 1, data = features)

out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# añadir jitter muy pequeño para evitar distancias cero
set.seed(123)
eps <- 1e-8
X_j <- X + matrix(rnorm(nrow(X)*ncol(X), sd = eps),
                  nrow = nrow(X))

#-------------------------------
# Paso 2: Sammon Mapping
#-------------------------------
d0  <- dist(X_j)                  # distancia original
sm  <- do.sammon(X_j, ndim = 2)   # Sammon Mapping
emb <- sm$Y                       # coordenadas de salida

#-------------------------------
# Paso 3: Métricas, guardar y plot
#-------------------------------
d1       <- dist(emb)
spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
ari      <- adjustedRandIndex(
               kmeans(emb, centers = length(unique(labels)))$cluster,
               labels
             )
silhouette_s <- mean(
                  silhouette(
                    kmeans(emb, centers = length(unique(labels)))$cluster,
                    d1
                  )[, "sil_width"]
                )
rank0    <- apply(as.matrix(d0), 2, rank, ties.method = "average")
rank1    <- apply(as.matrix(d1), 2, rank, ties.method = "average")
k        <- 7
trust_sm <- compute_trustworthiness(rank0, rank1, k)
cont_sm  <- compute_continuity(rank0, rank1, k)

dir.create("results", showWarnings = FALSE)
cat(sprintf(
  "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
  spearman, ari, silhouette_s, trust_sm, cont_sm
), file = "results/metrics_config_sammon_mushroom.txt")

# Plot
sam_df <- data.frame(Dim1 = emb[,1],
                     Dim2 = emb[,2],
                     Label = factor(labels))

p <- ggplot(sam_df, aes(Dim1, Dim2, color = Label)) +
  geom_point(size = 1) +
  ggtitle("Sammon Mapping") +
  big_text_theme

print(p)
dir.create("plots", showWarnings = FALSE)
ggsave("plots/Sammon_Mapping_mushroom_plot.png",
       plot = p, width = 8, height = 6, dpi = 300)
