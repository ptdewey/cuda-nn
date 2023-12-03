library(data.table)
library(dplyr)
library(ggplot2)
library(glue)
library(wesanderson)

# Update dir as necessary
dir <- "roofline_metrics"

read_metrics <- function(file) {
    version <- sub('.*v(.*?)\\.csv$', 'v\\1', file)
    fread(file) %>%
        filter(
            # `Metric Name` == "dram_read_throughput" |
            # `Metric Name` == "dram_write_throughput" |
            `Metric Name` == "dram_read_bytes" |
            `Metric Name` == "dram_write_bytes" |
            `Metric Name` == "flop_count_sp") %>%
        mutate(Version = version) %>%
        mutate(kernel = gsub("\\(.*?\\)", "", Kernel)) %>%
        rename(metric = `Metric Name`) %>%
        return()
}

read_metrics_dir <- function(dir) {
    files <- list.files(path = dir, full.names = TRUE, pattern = "*.csv")
    lapply(glue("{files}"), read_metrics) %>%
        bind_rows() %>%
        return()
}

# Read in all metrics csv files
df <- read_metrics_dir(dir)

calc_ai <- function(df, k) {
    temp <- df %>% 
        filter(kernel == k)
    out <- data.table(
        version = character(),
        arithmetic_intensity = numeric()
    )
    print(k)
    for (v in temp$Version) {
        t <- filter(temp, Version == v)
        ai <- as.numeric(t$Avg[1]) / (as.numeric(t$Avg[2]) + as.numeric(t$Avg[3]))
        out <- rbind(out, list(version = v, arithmetic_intensity = ai))
    }
    return(out)
}

calc_ai(df, "softmaxActivationBackprop")
calc_ai(df, "softmaxActivationForward")
calc_ai(df, "meanSquareErrorCost")
calc_ai(df, "dMeanSquareErrorCost")
calc_ai(df, "binaryCrossEntropyCost")
calc_ai(df, "dBinaryCrossEntropyCost")
calc_ai(df, "linearLayerForward")
calc_ai(df, "linearLayerBackprop")
calc_ai(df, "linearLayerUpdateWeights")
calc_ai(df, "linearLayerUpdateBias")

