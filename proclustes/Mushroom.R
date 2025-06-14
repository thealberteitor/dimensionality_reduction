library(Rtsne)
library(smacofx)
library(Rdimtools)
library(smacof)

source("../preprocess.R")
source("../plot_theme.R")

# Carga y preprocesado del dataset Mushroom 

col_names <- c(
  "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
  "gill_attachment", "gill_spacing", "gill_size", "gill_color",
  "stalk_shape", "stalk_root", "stalk_surface_above_ring",
  "stalk_surface_below_ring", "stalk_color_above_ring",
  "stalk_color_below_ring", "veil_type", "veil_color", "ring_number",
  "ring_type", "spore_print_color", "population", "habitat"
)


mushroom <- read.csv(
  "../Mushroom/data/mushroom.csv",
  header = FALSE,
  col.names = col_names,
  stringsAsFactors = TRUE
)

constant_cols <- sapply(mushroom, function(col) is.factor(col) && nlevels(col) < 2)
mushroom <- mushroom[, !constant_cols]

labels_raw <- mushroom$class
features <- mushroom[, setdiff(names(mushroom), "class")]
X_raw <- model.matrix(~ . - 1, data = features)


out <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X <- out$data

set.seed(123)
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))
d0 <- dist(X_jitter)


#Ejecutar todos los métodos de reducción de dimensionalidad 

cat("Ejecutando los 4 métodos de reducción de dimensionalidad para el dataset Mushroom...\n")

tsne_fit <- Rtsne(X_jitter, perplexity = 30, verbose = FALSE, max_iter = 1000)
emb_tsne <- tsne_fit$Y

lmds_fit <- lmds(delta = d0, k = 15, ndim = 2, tau = 0.5, verbose = 0)
emb_lmds <- lmds_fit$conf

sammon_fit <- Rdimtools::do.sammon(X_jitter, ndim = 2)
emb_sammon <- sammon_fit$Y

cmds_fit <- stats::cmdscale(d0, k = 2)
emb_cmds <- cmds_fit


# Normalización de los Embeddings

cat("Normalizando las configuraciones para el análisis de Procrustes...\n")

embeddings_list <- list(
  "t-SNE"       = emb_tsne,
  "MDS Local"   = emb_lmds,
  "Sammon"      = emb_sammon,
  "MDS Clásico" = emb_cmds
)

normalized_embeddings_list <- lapply(embeddings_list, function(mat) {
  mat_centered <- scale(mat, center = TRUE, scale = FALSE)
  sum_of_squares <- sum(mat_centered^2)
  if (sum_of_squares > 1e-9) { 
      mat_scaled <- mat_centered / sqrt(sum_of_squares)
  } else {
      mat_scaled <- mat_centered
  }
  return(mat_scaled)
})


# Análisis de Procrustes con datos normalizados

cat("Calculando la matriz de disimilitud de Procrustes...\n\n")

method_names <- names(normalized_embeddings_list)
procrustes_matrix <- matrix(NA, nrow = 4, ncol = 4, dimnames = list(method_names, method_names))

for (i in seq_along(method_names)) {
  for (j in seq_along(method_names)) {
    if (i == j) {
      procrustes_matrix[i, j] <- 0
      next
    }
    embedding_A <- normalized_embeddings_list[[i]]
    embedding_B <- normalized_embeddings_list[[j]]
    
    procrustes_fit <- Procrustes(X = embedding_A, Y = embedding_B)
    procrustes_matrix[i, j] <- sum((procrustes_fit$X - procrustes_fit$Yhat)^2)
  }
}


cat("------------------------------------------------------------\n")
cat("Matriz de Disimilitud de Procrustes (m^2) para el Dataset Mushroom (Corregida):\n")
cat("------------------------------------------------------------\n\n")
print(round(procrustes_matrix, 3))


dir.create("results", showWarnings = FALSE)

output_file <- "results/procrustes_matrix_mushroom.txt"

write.table(
  round(procrustes_matrix, 3),
  file = output_file,
  sep = "\t",         
  quote = FALSE,      
  col.names = NA   
)

cat(paste("\nLa matriz de resultados ha sido guardada en:", output_file, "\n"))
