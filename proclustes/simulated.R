
library(Rtsne)
library(smacofx)
library(Rdimtools)
library(smacof)


source("../plot_theme.R")

# Carga y preprocesado del dataset Sintético 
sim_data <- readRDS("../Simulated/simulated_data_list.rds")


X <- sim_data$data_scaled
labels <- sim_data$labels

set.seed(123)
eps <- 1e-8
X_jitter <- X + matrix(rnorm(nrow(X) * ncol(X), sd = eps), nrow = nrow(X))
d0 <- dist(X_jitter)


# Ejecutar TODAS las configuraciones de los métodos
cat("Ejecutando 15 configuraciones de algoritmos...\n\n")


embeddings_list <- list()


perplexities <- c(5, 15, 30, 50)
for (p in perplexities) {
  cat(sprintf("Ejecutando t-SNE con perplexity = %d...\n", p))
  method_name <- sprintf("t-SNE (p=%d)", p)
  tsne_fit <- Rtsne(X_jitter, perplexity = p, verbose = FALSE, max_iter = 200, check_duplicates = FALSE)
  embeddings_list[[method_name]] <- tsne_fit$Y
}

k_vals <- c(5, 15, 30)
tau_vals <- c(0.3, 0.5, 0.7)
for (k in k_vals) {
  for (tau in tau_vals) {
    cat(sprintf("Ejecutando MDS Local con k = %d y tau = %.1f...\n", k, tau))
    method_name <- sprintf("MDS Local (k=%d, tau=%.1f)", k, tau)
    # verbose=0 para no llenar la consola
    lmds_fit <- lmds(delta = d0, k = k, ndim = 2, tau = tau, verbose = 0) 
    embeddings_list[[method_name]] <- lmds_fit$conf
  }
}

cat("Ejecutando Sammon Mapping...\n")
sammon_fit <- Rdimtools::do.sammon(X_jitter, ndim = 2)
embeddings_list[["Sammon"]] <- sammon_fit$Y

cat("Ejecutando MDS Clásico...\n")
cmds_fit <- stats::cmdscale(d0, k = 2)
embeddings_list[["MDS Clásico"]] <- cmds_fit


# Normalización de TODOS los Embeddings

cat("\nNormalizando las 15 configuraciones para el análisis de Procrustes...\n")

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


# Análisis de Procrustes

cat("Calculando la matriz de disimilitud de Procrustes (15x15)...\n\n")

method_names <- names(normalized_embeddings_list)
procrustes_matrix <- matrix(NA, nrow = length(method_names), ncol = length(method_names),
                           dimnames = list(method_names, method_names))
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
cat("Matriz de Disimilitud de Procrustes Extendida para Dataset Sintético:\n")
cat("------------------------------------------------------------\n\n")
print(round(procrustes_matrix, 3))


dir.create("results", showWarnings = FALSE)
output_file <- "results/procrustes_matrix_simulated_extended.txt"

write.table(
  round(procrustes_matrix, 3),
  file = output_file,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

cat(paste("\nLa matriz de resultados ha sido guardada en:", output_file, "\n"))
