# Imputación de NA por mediana de su clase
impute_by_class_median <- function(data, labels) {
  if (anyNA(data)) {
    for (j in seq_len(ncol(data))) {
      data[, j] <- ave(
        data[, j], labels,
        FUN = function(x) {
          x[is.na(x)] <- median(x, na.rm = TRUE)
          x
        }
      )
    }
  }
  data
}

# Eliminación de columnas de varianza cero
# Para evitar divisiones por cero en el escalado
drop_constant_columns <- function(data) {
  non_const <- apply(data, 2, function(col) sd(col, na.rm = TRUE) > 0)
  data[, non_const, drop = FALSE]
}

# Eliminación de outliers por Z‑score
# Devuelve la matriz limpia y el vector lógico de filas retenidas
enable_outlier_removal <- function(data, threshold = 3) {
  z <- scale(data)
  keep <- apply(z, 1, function(row) all(abs(row) < threshold))
  list(data = data[keep, , drop = FALSE], keep = keep)
}

# Normalización final (media 0, varianza 1)
normalize_data <- function(data) {
  scale(data)
}



preprocess_data <- function(data, labels, z_threshold = 3) {
  data_imp   <- impute_by_class_median(data, labels)
  data_cc1   <- drop_constant_columns(data_imp)
  out        <- enable_outlier_removal(data_cc1, z_threshold)
  data_no    <- out$data
  keep       <- out$keep

  data_cc2   <- drop_constant_columns(data_no)
  data_norm  <- normalize_data(data_cc2)
  list(
    data   = data_norm,
    labels = labels[keep]
  )
}
