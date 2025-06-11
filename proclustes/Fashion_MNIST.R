library(Rtsne)
library(smacofx)
library(Rdimtools)
library(smacof)
library(keras)

# Asumiendo que los scripts locales están en la ruta correcta
source("../preprocess.R")
source("../plot_theme.R")


# --- PASO 2: Carga y preprocesado del dataset Fashion-MNIST ---

# Cargar el dataset Fashion-MNIST
fashion_mnist <- dataset_fashion_mnist()
set.seed(123)

n_samples <- 20000
idx        <- sample(seq_len(nrow(fashion_mnist$train$x)), n_samples)
data_raw   <- fashion_mnist$train$x[idx,,, drop = FALSE]
labels_raw <- fashion_mnist$train$y[idx]

# Preprocesado específico para Fashion-MNIST
X_raw <- array_reshape(data_raw, c(n_samples, 28*28)) / 255
out    <- preprocess_data(X_raw, labels_raw, z_threshold = 3)
X      <- out$data

set.seed(123)
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))

# Nota: Calcular la matriz de distancias para 20,000 puntos es muy intensivo
cat("Calculando la matriz de distancias para Fashion-MNIST... \n")
d0 <- dist(X_jitter)


# --- PASO 3: Ejecutar todos los métodos de reducción de dimensionalidad ---

cat("Ejecutando los 4 métodos de reducción de dimensionalidad para el dataset Fashion-MNIST...\n")

# Los parámetros se mantienen idénticos para una comparación justa
# a) t-SNE
tsne_fit <- Rtsne(X_jitter, perplexity = 30, verbose = FALSE, max_iter = 1000, check_duplicates = FALSE)
emb_tsne <- tsne_fit$Y

# b) MDS Local
lmds_fit <- lmds(delta = d0, k = 15, ndim = 2, tau = 0.5, verbose = 1)
emb_lmds <- lmds_fit$conf

# c) Sammon Mapping
sammon_fit <- Rdimtools::do.sammon(X_jitter, ndim = 2)
emb_sammon <- sammon_fit$Y

# d) MDS Clásico (especificando el paquete 'stats' para evitar conflictos)
cmds_fit <- stats::cmdscale(d0, k = 2)
emb_cmds <- cmds_fit


# --- PASO 4: Normalización de los Embeddings (CRUCIAL) ---

cat("Normalizando las configuraciones para el análisis de Procrustes...\n")

embeddings_list <- list(
  "t-SNE"       = emb_tsne,
  "MDS Local"   = emb_lmds,
  "Sammon"      = emb_sammon,
  "MDS Clásico" = emb_cmds
)

# Aplicar la normalización (centrado y escalado a suma de cuadrados = 1)
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


# --- PASO 5: Análisis de Procrustes con datos normalizados ---

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


# --- PASO 6: Mostrar y guardar la Matriz de Disimilitud Resultante ---

cat("------------------------------------------------------------\n")
cat("Matriz de Disimilitud de Procrustes (m^2) para el Dataset Fashion-MNIST (Corregida):\n")
cat("------------------------------------------------------------\n\n")
print(round(procrustes_matrix, 3))

# --- PASO 7: Guardar la matriz en un fichero de resultados ---

dir.create("results", showWarnings = FALSE)
output_file <- "results/procrustes_matrix_fashion_mnist.txt"

write.table(
  round(procrustes_matrix, 3),
  file = output_file,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

cat(paste("\nLa matriz de resultados ha sido guardada en:", output_file, "\n"))
