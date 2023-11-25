library(data.table)
library(dplyr)
library(ggplot2)
library(glue)
library(purrr)
library(tidyr)
library(wesanderson)

colors <- wes_palette("AsteroidCity1")

### Timing
trace <- "profiler/profiler_trace_out_v"
v0_time <- fread(glue("{trace}0.csv"))
v1_time <- fread(glue("{trace}1.csv"))
v2_time <- fread(glue("{trace}2.csv"))
v3_time <- fread(glue("{trace}3.csv"))
tv0 <- sum(as.numeric(v0_time$Duration), na.rm = TRUE)
tv1 <- sum(as.numeric(v1_time$Duration), na.rm = TRUE)
tv2 <- sum(as.numeric(v2_time$Duration), na.rm = TRUE)
tv3 <- sum(as.numeric(v3_time$Duration), na.rm = TRUE)

print(paste("Execution time of v0 kernel:", tv0))
print(paste("Execution time of v1 kernel:", tv1))
print(paste("Execution time of v2 kernel:", tv2))
print(paste("Execution time of v3 kernel:", tv3))

# NOTE: update with later versions
times <- data.frame(version = c("v0", "v1", "v2", "v3"),
                    execution_time = c(tv0, tv1, tv2, tv3))

plot <- FALSE

# TODO: update epoch number as necessary
if (plot == TRUE) {
    p <- ggplot(times, aes(x = version, y = execution_time / 1000)) +
      geom_point(size = 3, color = colors[3]) +
      geom_line(aes(group = 1), color = colors[3]) +
      ggtitle("Fifteen Epoch Execution Time Per Version") +
      ylab("Execution Time (milliseconds)") +
      xlab("Version")
    ggsave("images/version_timing.png", p)
}

# kernel by kernel breakdown of timings
# remove non-kernel information from csvs
extract_kernels <- function(df) {
  df$Duration <- as.numeric(df$Duration)
  df <- df %>%
    filter(!grepl("\\[|\\]", Name)) %>%
    group_by(Name) %>%
    summarize(
      # min_time = min(Duration),
      # max_time = max(Duration),
      avg_time = mean(Duration)
    )
  df[-1, ]
}

v0t <- extract_kernels(v0_time)
v1t <- extract_kernels(v1_time)
v2t <- extract_kernels(v2_time)
v3t <- extract_kernels(v3_time)
vs <- list(v0t, v1t, v2t, v3t)

v_all <- imap_dfr(vs, ~ mutate(.x, version = glue("v{.y - 1}")))

print(v_all)


plot_kernel <- function(df, name) {
  df <- df %>%
    filter(grepl(name, Name))

  if (nrow(df) == 0) {
    warning("No partial match found.")
    return(NULL)
  }

  p <- ggplot(df, aes(x = version, y = avg_time)) +
    geom_point(size = 3, color = colors[1]) +
    geom_line(aes(group = 1), linetype = "solid", color = colors[1]) +
    labs(title = glue("Execution Time Progression for {name} Kernel"),
         x = "Kernel Version", y = "Average Time (microseconds)")
  ggsave(glue("images/{name}.png"), p)
}

# Plots
if (plot == TRUE) {
  for (name in unique(v_all$Name)) {
    name <- gsub("\\(.*\\)", "", name)
    p <- plot_kernel(v_all, name)
  }
}

# good table for report - (remove min/max first?)
v_tab <- v_all %>% pivot_wider(names_from = "version", values_from = "avg_time")
print(v_tab)
