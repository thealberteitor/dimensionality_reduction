# --- generate_simulated_data.R ---

# 0. Cargar paquetes necesarios
library(MASS)

# 1. Definir Parámetros Globales de la Simulación
set.seed(123)
N_points_per_cluster <- 100
N_clusters <- 3
D_alta <- 50
noise_sd <- 0.1
output_file_rds <- "simulated_data_list.rds" # Nombre del archivo para guardar

# 2. Función para Generar Datos (la misma que antes)
generate_gaussian_clusters <- function(n_points_k, n_k, D_high, separation_factor = 5, base_sd = 1) {
  all_data <- data.frame()
  true_labels <- c()
  for (k in 1:n_k) {
    mean_vector <- rep(0, D_high)
    mean_vector[k] <- separation_factor * (k - 1)
    cluster_data <- mvrnorm(n = n_points_k, mu = mean_vector, Sigma = diag(base_sd^2, D_high))
    all_data <- rbind(all_data, cluster_data)
    true_labels <- c(true_labels, rep(k, n_points_k))
  }
  if (noise_sd > 0) {
      noise_matrix <- matrix(rnorm(nrow(all_data) * D_high, 0, noise_sd), ncol = D_high)
      all_data <- all_data + noise_matrix
  }
  colnames(all_data) <- paste0("Dim", 1:D_high)
  return(list(data = as.matrix(all_data), labels = factor(true_labels)))
}

# Generar datos
sim_data_list <- generate_gaussian_clusters(N_points_per_cluster, N_clusters, D_alta, separation_factor = 10)

# (Opcional pero recomendado) Escalar los datos ANTES de guardarlos si siempre los vas a usar escalados
# O escalar DESPUÉS de cargarlos en el script de análisis.
# Por consistencia, hagámoslo aquí, pero es una elección de flujo de trabajo.
sim_data_list$data_scaled <- scale(sim_data_list$data) 

# Guardar la lista completa en un archivo .rds
saveRDS(sim_data_list, file = output_file_rds)

print(paste("Datos simulados generados. Dimensiones (original):", 
            nrow(sim_data_list$data), "x", ncol(sim_data_list$data)))
print(paste("Datos guardados en:", output_file_rds))