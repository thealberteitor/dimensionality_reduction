#!/usr/bin/env Rscript

# local_mds_mushroom_fixed.R (Local MDS con smacofx corregido)
library(smacofx)
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
  "class",
  "cap_shape","cap_surface","cap_color","bruises","odor",
  "gill_attachment","gill_spacing","gill_size","gill_color",
  "stalk_shape","stalk_root",
  "stalk_surface_above_ring","stalk_surface_below_ring",
  "stalk_color_above_ring","stalk_color_below_ring",
  "veil_type","veil_color","ring_number","ring_type",
  "spore_print_color","population","habitat"
)
mushroom <- read.csv(
  "data/mushroom.csv",
  header = FALSE,
  col.names = col_names,
  stringsAsFactors = TRUE
)

constant_cols <- sapply(mushroom, function(col)
  is.factor(col) && nlevels(col) < 2
)
mushroom <- mushroom[, !constant_cols]

labels_raw <- mushroom$class
features   <- mushroom[, setdiff(names(mushroom), "class")]
X_raw      <- model.matrix(~ . - 1, data = features)

out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data
labels <- out$labels

# --- Inicio de Diagnóstico Previo (ya implementado por ti) ---
cat(sprintf("Dimensions of X after preprocess_data: %d rows, %d columns\n", nrow(X), ncol(X)))
cat(sprintf("Number of labels after preprocess_data: %d\n", length(labels)))

if (nrow(X) < 20) {
  warning("PREPROCESSING WARNING: Very few samples remain after preprocessing. This could be the cause of issues.")
}
# --- Fin de Diagnóstico Previo ---

set.seed(123)
eps      <- 1e-8
X_jitter <- X + matrix(
  rnorm(nrow(X) * ncol(X), sd = eps),
  nrow = nrow(X)
)

#-------------------------------
# Paso 2: Local MDS con smacofx
#-------------------------------
d0      <- dist(X_jitter)
d0_mat  <- as.matrix(d0)

k   <- 15
tau <- 0.5

# Comprobación de que d0_mat no tenga NAs o Infs si X es muy pequeño
if(any(!is.finite(d0_mat))) {
  stop("ERROR: Distance matrix d0_mat contains non-finite values. Check input X_jitter.")
}
if(nrow(X) < k) {
  warning(sprintf("Warning: Number of samples (%d) is less than k (%d). This might affect lmds.", nrow(X), k))
}


cat("Starting LMDS minimization...\n") # Añadido para ver si llega aquí
res_lmds <- lmds(
  delta   = d0_mat,
  k       = k,
  ndim    = 2,
  tau     = tau,
  verbose = 1
)
emb <- res_lmds$conf
cat("LMDS minimization finished.\n") # Añadido para ver si termina

#-------------------------------
# Paso 3: Métricas, guardar y plot
#-------------------------------
emb_cent <- scale(emb, center = TRUE, scale = FALSE)

# --- Inicio de NUEVO Diagnóstico ---
cat(sprintf("Dimensions of emb_cent (embedding coordinates): %d rows, %d columns\n", nrow(emb_cent), ncol(emb_cent)))

if (nrow(emb_cent) > 0) {
  # Para contar puntos únicos, redondeamos para manejar imprecisiones de punto flotante
  rounded_emb_cent <- round(emb_cent, digits = 5)
  num_unique_points <- nrow(unique(rounded_emb_cent))
  cat(sprintf("Number of unique 2D points in the embedding: %d\n", num_unique_points))

  if (num_unique_points < 20 && nrow(emb_cent) > num_unique_points) { # Si hay muchos más puntos de entrada que de salida únicos
    warning("EMBEDDING WARNING: The LMDS embedding has collapsed many input points into very few unique 2D locations!")
    cat("First few unique points (rounded):\n")
    print(head(unique(rounded_emb_cent)))
  }
} else {
  warning("EMBEDDING WARNING: emb_cent has 0 rows. LMDS might have failed or returned an empty configuration.")
}
# --- Fin de NUEVO Diagnóstico ---


d1       <- dist(emb_cent)

# Verificar que d1 no sea problemático si emb_cent es muy pequeño
if (length(as.vector(d1)) == 0 && length(as.vector(d0)) > 0) {
    warning("Metrics WARNING: d1 is empty, possibly due to emb_cent having too few points for dist(). Metrics might be NA.")
    spearman <- NA
    ari <- NA
    silhouette_avg <- NA
    trust <- NA
    continuity <- NA
} else if (length(unique(labels)) < 2 && nrow(emb_cent) >= 2) {
    warning("Metrics WARNING: Fewer than 2 unique labels. ARI and Silhouette might not be meaningful or might error.")
    spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
    ari <- NA # kmeans needs at least 2 centers, unique(labels) would be 1
    silhouette_avg <- NA # silhouette needs clusters
    rank0 <- apply(d0_mat, 2, rank, ties.method = "average")
    rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average")
    trust     <- compute_trustworthiness(rank0, rank1, k)
    continuity <- compute_continuity(rank0, rank1, k)
} else if (nrow(emb_cent) < length(unique(labels)) || nrow(emb_cent) < 2) {
    warning("Metrics WARNING: Not enough points in embedding for kmeans or silhouette. Skipping some metrics.")
    spearman <- if(length(as.vector(d0)) > 0 && length(as.vector(d1)) > 0) cor(as.vector(d0), as.vector(d1), method = "spearman") else NA
    ari <- NA
    silhouette_avg <- NA
    trust <- NA
    continuity <- NA
} else {
    spearman <- cor(as.vector(d0), as.vector(d1), method = "spearman")
    num_clusters_kmeans <- length(unique(labels))
    kmeans_result <- kmeans(emb_cent, centers = num_clusters_kmeans)
    
    ari <- adjustedRandIndex(
      kmeans_result$cluster,
      labels
    )

    silhouette_avg <- mean(
      silhouette(
        kmeans_result$cluster,
        d1 # d1 es dist(emb_cent)
      )[, "sil_width"]
    )

    rank0 <- apply(d0_mat, 2, rank, ties.method = "average")
    rank1 <- apply(as.matrix(d1), 2, rank, ties.method = "average") # d1 es objeto dist, convertir a matriz para apply
    trust     <- compute_trustworthiness(rank0, rank1, k)
    continuity <- compute_continuity(rank0, rank1, k)
}


dir.create("results", showWarnings = FALSE)
cat(sprintf(
  "Spearman=%.3f\nARI=%.3f\nSilhouette=%.3f\nTrustworthiness=%.3f\nContinuity=%.3f\n",
  spearman, ari, silhouette_avg, trust, continuity
), file = "results/metrics_config_local_mds_mushroom.txt")


# Solo intentar plotear si hay puntos
if (nrow(emb_cent) > 0) {
  mds_df <- data.frame(
    Dim1  = emb_cent[,1],
    Dim2  = emb_cent[,2],
    Label = factor(labels) # Asegurar que labels corresponda a emb_cent en longitud
  )

  p <- ggplot(mds_df, aes(x = Dim1, y = Dim2, color = Label)) +
    geom_point(size = 1, alpha = 0.5) + # Añadido alpha para transparencia
    ggtitle("MDS Local") +
    big_text_theme

  print(p)

  dir.create("plots", showWarnings = FALSE)
  ggsave(
    "plots/Local_MDS_mushroom_plot.png",
    plot = p,
    width = 8,
    height = 6,
    dpi = 300
  )
} else {
  cat("PLOTTING INFO: No points to plot as emb_cent is empty or has too few rows.\n")
  # Crear un archivo de texto indicando que no se generó el plot
  cat("Plot not generated due to insufficient data after LMDS.",
      file = "plots/Local_MDS_mushroom_plot_nodata.txt")
}

cat("Script finished.\n")