library(ggplot2)

big_text_theme <- theme_minimal(base_size = 16) +
  theme(
    plot.title   = element_text(size = 24, hjust = 0.5),
    axis.title   = element_text(size = 20),
    axis.text    = element_text(size = 18),
    legend.title = element_text(size = 20),
    legend.text  = element_text(size = 18)
  )

theme_set(big_text_theme)
