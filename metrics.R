compute_trustworthiness <- function(rank_original, rank_embedding, k) {
  n <- nrow(rank_original)
  total <- 0
  for (i in seq_len(n)) {
    nn_embed <- which(rank_embedding[, i] >= 2 & rank_embedding[, i] <= k + 1)
    U_k_i <- nn_embed[ rank_original[nn_embed, i] > k + 1 ]
    penalties <- rank_original[U_k_i, i] - k
    total <- total + sum(penalties)
  }
  1 - (2 / (n * k * (2*n - 3*k - 1))) * total
}



compute_continuity <- function(rank_original, rank_embedding, k) {
  n <- nrow(rank_original)
  total <- 0
  for (i in seq_len(n)) {
    nn_orig <- which(rank_original[, i] >= 2 & rank_original[, i] <= k + 1)
    V_k_i <- nn_orig[ rank_embedding[nn_orig, i] > k + 1 ]
    penalties <- rank_embedding[V_k_i, i] - k
    total <- total + sum(penalties)
  }
  1 - (2 / (n * k * (2*n - 3*k - 1))) * total
}
