## Monitoring the log-likelihood across iterations for MWG and PG
library(reshape2)
library(ggplot2)

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/final_analysis/beta2_epochs250/")

## pseudo-gibbs
pg_files <- list.files("log_likelihood/PG")
for (i in 1:length(pg_files)) {
  df <- read.csv(paste0("log_likelihood/PG/",pg_files[i]), header = F)
  datname <- gsub("loglikelihood_across_iterations_","",gsub(".csv","",pg_files[i]))
  colnames(df) <- datname
  
  if (i==1) {
    pg_final <- df
  } else {
    pg_final <- cbind(pg_final, df)
  }
  
  rm(df)
}

pg_final$iteration <- as.character(1:nrow(pg_final))
pg_final_melt <- melt(pg_final, by = "iteration")
pg_final_melt$iteration <- as.numeric(pg_final_melt$iteration)

pg_trace <- ggplot(data = pg_final_melt[pg_final_melt$variable=="plaus_dataset_1" & pg_final_melt$iteration %in% 1:100,], mapping = aes(x = iteration, y = value)) +
  geom_point()  +
  geom_line() +
  theme_bw() +
  ylab("Log likelihood") +
  xlab("Iteration")

## MWG
mwg_files <- list.files("log_likelihood/MWG")
for (i in 1:length(mwg_files)) {
  df <- read.csv(paste0("log_likelihood/MWG/",mwg_files[i]), header = F)
  datname <- gsub("loglikelihood_across_iterations_","",gsub(".csv","",mwg_files[i]))
  colnames(df) <- datname
  
  if (i==1) {
    mwg_final <- df
  } else {
    mwg_final <- cbind(mwg_final, df)
  }
  
  rm(df)
}

mwg_final$iteration <- as.character(1:nrow(mwg_final))
mwg_final_melt <- melt(mwg_final, by = "iteration")
mwg_final_melt$iteration <- as.numeric(mwg_final_melt$iteration)

mwg_trace <- ggplot(data = mwg_final_melt[mwg_final_melt$variable=="plaus_dataset_1" & mwg_final_melt$iteration %in% 1:100,], mapping = aes(x = iteration, y = value)) +
  geom_point()  +
  geom_line() +
  theme_bw() +
  ylab("Log likelihood") +
  xlab("Iteration")

si <- read.csv("log_likelihood/loglikelihood_across_iterations_single_imputed_dataset.csv", header = F)

si$iteration <- 1:nrow(si)

si_trace <- ggplot(data = si[si$iteration %in% 1:100,], mapping = aes(x = iteration, y = V1)) +
  geom_point()  +
  geom_line() +
  theme_bw() +
  ylab("Log likelihood") +
  xlab("Iteration")

## Export plots
pdf("../../ICLR_paper/figures/loglikelihood_traceplot_MWG.pdf", 3, 3)
print(mwg_trace)
dev.off()

pdf("../../ICLR_paper/figures/loglikelihood_traceplot_PG.pdf", 3, 3)
print(pg_trace)
dev.off()

pdf("../../ICLR_paper/figures/loglikelihood_traceplot_SI.pdf", 3, 3)
print(si_trace)
dev.off()
