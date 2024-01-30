library(xtable)
library(data.table)
library(dplyr)
library(ggplot2)
library(glue)
library(purrr)
library(tidyr)
library(wesanderson)

colors <- wes_palette("AsteroidCity1")

### Timing
read_trace <- function(version) {
  # df <- fread(glue("profiler2/profiler_trace_out_v{version}.csv"))
  df <- fread(glue("cost_improvement/profiler_trace_out_v{version}.csv"))

  # fix mis-matched timescales (ms and mus)
  if (df[1, ]$Duration == "ms") {
    df <- df[-1, ]
    df$Duration <- as.numeric(df$Duration) * 1000
    return(df)
  }

  # remove units column
  df <- df[-1, ]
  return(df)
}

v0_time <- read_trace(0)
v1_time <- read_trace(1)
v2_time <- read_trace(2)
v3_time <- read_trace(3)
v4_time <- read_trace(4)
v6_time <- read_trace(6)


tv0 <- sum(as.numeric(v0_time$Duration), na.rm = TRUE)
tv1 <- sum(as.numeric(v1_time$Duration), na.rm = TRUE)
tv2 <- sum(as.numeric(v2_time$Duration), na.rm = TRUE)

tv3 <- sum(as.numeric(v3_time$Duration), na.rm = TRUE)
tv4 <- sum(as.numeric(v4_time$Duration), na.rm = TRUE)
tv6 <- sum(as.numeric(v6_time$Duration), na.rm = TRUE)


# NOTE: update with later versions
times <- data.frame(version = c("v0", "v1", "v2", "v3", "v4", "v6"),
    execution_time = c(tv0, tv1, tv2, tv3, tv4, tv6))

plot <- TRUE
# plot <- FALSE

# TODO: update epoch number as necessary
if (plot == TRUE) {
    p <- ggplot(times, aes(x = version, y = execution_time / 1000)) +
    geom_point(size = 3, color = colors[3]) +
    geom_line(aes(group = 1), color = colors[3]) +
    ggtitle("1 Epoch Execution Time Per Version") +
    ylab("Execution Time (milliseconds)") +
    xlab("Version")
    ggsave("images/version_timing.png", p)
}

# kernel by kernel breakdown of timings
# remove non-kernel information from csvs
extract_kernels <- function(df) {
    df$Duration <- as.numeric(df$Duration)
    df <- df %>%
        mutate(Name = gsub("\\(.*?\\)", "", Name)) %>%
        filter(!grepl("\\[|\\]", Name)) %>%
        group_by(Name) %>%
        summarize(
            avg_time = mean(Duration)
        )
}

v0t <- extract_kernels(v0_time)
v1t <- extract_kernels(v1_time)
v2t <- extract_kernels(v2_time)
vs_bin <- imap_dfr(list(v0t, v1t, v2t), ~ mutate(.x, version = glue("v{.y - 1}")))

v3t <- extract_kernels(v3_time)
v4t <- extract_kernels(v4_time)
v6t <- extract_kernels(v6_time)
vs <- list(v0t, v1t, v2t, v3t, v4t, v6t)
vs_mul <- imap_dfr(list(v3t, v4t, v6t), ~ mutate(.x, version = glue("v{.y + 2}")))
idc <- which(vs_mul$version == "v5")
vs_mul$version[idc] <- rep_len("v6", length(idc))


# v_all <- imap_dfr(vs, ~ mutate(.x, version = glue("v{.y - 1}")))
# print(v_all)


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
  for (name in unique(vs_bin$Name)) {
    name <- gsub("\\(.*\\)", "", name)
    p <- plot_kernel(vs_bin, name)
  }
  for (name in unique(vs_mul$Name)) {
    name <- gsub("\\(.*\\)", "", name)
    p <- plot_kernel(vs_mul, name)
  }
}

print(paste("Single epoch execution time of v0 (s):", tv0 / 1000000))
print(paste("Single epoch execution time of v1 (s):", tv1 / 1000000))
print(paste("Single epoch execution time of v2 (s):", tv2 / 1000000))
print(paste("Single epoch execution time of v3 (s):", tv3 / 1000000))
print(paste("Single epoch execution time of v4 (s):", tv4 / 1000000))
print(paste("Single epoch execution time of v6 (s):", tv6 / 1000000))

# good table for report - (remove min/max first?)
v_bin_tab <- vs_bin %>%
  pivot_wider(names_from = "version", values_from = "avg_time")
print(v_bin_tab)
xtable(v_bin_tab)

v_mul_tab <- vs_mul %>%
  pivot_wider(names_from = "version", values_from = "avg_time")
print(v_mul_tab)
xtable(v_mul_tab)
