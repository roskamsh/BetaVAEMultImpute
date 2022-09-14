## Compare percentiles from different beta and epochs numbers
library(reshape2)
library(ggplot2)
library(cowplot)

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/final_analysis/compare_percentiles_different_training/")

training_runs <- c("beta1_epochs250","beta2_epochs250","beta2_epochs275", "beta2_epochs300", "beta2.5_epochs300")
imputation <- c("importance-sampling","metropolis-within-gibbs","pseudo-gibbs")

plots_percentiles <- list()
plots_deviation_percentiles <- list()

for (i in 1:length(training_runs)) {
  training_run <- training_runs[i]
  
  is <- read.csv(paste0("../", training_run, "/", imputation[1], "_imputation_percentiles.csv"))
  met <- read.csv(paste0("../", training_run, "/", imputation[2], "_imputation_percentiles.csv"))
  pg <- read.csv(paste0("../", training_run, "/", imputation[3], "_imputation_percentiles.csv"))
  
  res <- rbind(is, met, pg)
  res_melt <- melt(res, by = "X")
  res_melt$true_value <- c(rep(0.25, 3), rep(0.5, 3), rep(0.75, 3), rep(0.95, 3), rep(0.99, 3))
  
  p1 <- ggplot(data = res_melt, mapping = aes(x = true_value, y = value, colour = X)) +
    geom_point() +
    geom_abline(slope = 1, linetype = "dashed", colour = "dark grey") +
    theme_bw() +
    xlab("True percentile") +
    ylab("Percentile from imputed distribution") +
    ylim(c(0.2, 1)) +
    theme(legend.title = element_blank())
  
  plots_percentiles[[training_run]] <- p1
  
  res_melt$dev <- res_melt$value - res_melt$true_value
  
  p2 <- ggplot(data = res_melt, mapping = aes(x = variable, y = dev, colour = X)) +
    geom_point() +
    theme_bw() +
    ylab("Deviation from true value") +
    theme(legend.title = element_blank(),
          axis.title.x = element_blank()) +
    ylim(c(-0.15, 0.15))
  
  plots_deviation_percentiles[[training_run]] <- p2
}

library(cowplot)
pdf("Percentiles_all_model_params_GRID.pdf", width = 24, height = 4)
plot_grid(plotlist = plots_percentiles, labels = names(plots_percentiles), nrow = 1, label_size = 8)
dev.off()

pdf("Deviation_from_percentiles_all_model_params_GRID.pdf", width = 24, height = 4)
plot_grid(plotlist = plots_deviation_percentiles, labels = names(plots_deviation_percentiles), nrow = 1, label_size = 8)
dev.off()


