## Beta = 1 coverage and accuracy results

library(ggplot2)
library(dplyr)
library(reshape2)

setwd('/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/final_analysis/beta1_epochs250/')

# Read in data
na_vals_is <- read.csv("importance-sampling_compiled_NA_indices.csv", stringsAsFactors = F)
na_vals_mg <- read.csv("metropolis-within-gibbs_compiled_NA_indices.csv", stringsAsFactors = F)
na_vals_pg <- read.csv("pseudo-gibbs_compiled_NA_indices.csv", stringsAsFactors = F)

# Rank NA indices by true value and look at boxplot
# Highlight the plausible datasets that are outliers in the Intercept estimation
# Get data in right format for plotting
na_vals_is <- na_vals_is[order(na_vals_is$true_values),]
na_vals_is$NA_rank <- 1:nrow(na_vals_is)

## HERE - we need to subset this dataframe to include ~ 50 points, that are evenly distributed across the 1st, 2nd and 3rd third of the rank
dwnsmp <- sample(rownames(na_vals_is), 1000)
sub_na_vals_is <- na_vals_is[dwnsmp,]

sub_na_vals_is$NA_rank <- as.factor(sub_na_vals_is$NA_rank)
na_vals_is_melt <- melt(sub_na_vals_is, by = c("NA_rank"))
na_vals_is_melt$NA_rank <- as.numeric(paste(na_vals_is_melt$NA_rank))

f1a <- ggplot(data = na_vals_is_melt, mapping = aes(x = NA_rank, y = value)) +
  geom_point() +
  geom_point(data = na_vals_is_melt[na_vals_is_melt$variable=="true_values",], colour = "red") +
  theme_bw() +
  ylim(c(min(na_vals_is_melt$value) - 1, max(na_vals_is_melt$value + 1))) +
  xlab("NA index, ranked by true value") +
  ylab("Imputed value across \n100 plausible datasets")

## Now same for metropolis-within-gibbs and pseudo-gibbs
## Metropolis-within-Gibbs
na_vals_mg <- na_vals_mg[order(na_vals_mg$true_values),]
na_vals_mg$NA_rank <- 1:nrow(na_vals_mg)

## HERE - we need to subset this dataframe to include ~ 50 points, that are evenly distributed across the 1st, 2nd and 3rd third of the rank
sub_na_vals_mg <- na_vals_mg[dwnsmp,]

sub_na_vals_mg$NA_rank <- as.factor(sub_na_vals_mg$NA_rank)
na_vals_mg_melt <- melt(sub_na_vals_mg, by = c("NA_rank"))
na_vals_mg_melt$NA_rank <- as.numeric(paste(na_vals_mg_melt$NA_rank))

f1b <- ggplot(data = na_vals_mg_melt, mapping = aes(x = NA_rank, y = value)) +
  geom_point() +
  geom_point(data = na_vals_mg_melt[na_vals_mg_melt$variable=="true_values",], colour = "red") +
  theme_bw() +
  ylim(c(min(na_vals_is_melt$value) - 1, max(na_vals_is_melt$value + 1))) +
  xlab("NA index, ranked by true value") +
  ylab("Imputed value across \n100 plausible datasets")

## Pseudo-gibbs
na_vals_pg <- na_vals_pg[order(na_vals_pg$true_values),]
na_vals_pg$NA_rank <- 1:nrow(na_vals_pg)

## HERE - we need to subset this dataframe to include ~ 50 points, that are evenly distributed across the 1st, 2nd and 3rd third of the rank
sub_na_vals_pg <- na_vals_pg[dwnsmp,]

sub_na_vals_pg$NA_rank <- as.factor(sub_na_vals_pg$NA_rank)
na_vals_pg_melt <- melt(sub_na_vals_pg, by = c("NA_rank"))
na_vals_pg_melt$NA_rank <- as.numeric(paste(na_vals_pg_melt$NA_rank))

f1c <- ggplot(data = na_vals_pg_melt, mapping = aes(x = NA_rank, y = value)) +
  geom_point() +
  geom_point(data = na_vals_pg_melt[na_vals_pg_melt$variable=="true_values",], colour = "red") +
  theme_bw() +
  ylim(c(min(na_vals_is_melt$value) - 1, max(na_vals_is_melt$value + 1))) +
  xlab("NA index, ranked by true value") +
  ylab("Imputed value across \n100 plausible datasets")

## Now read in single imputation NA indices imputation
na_vals_single <- read.csv("NA_imputed_values_single_imputed_dataset.csv", stringsAsFactors = F, row.names = 1)

na_vals_single <- na_vals_single[order(na_vals_single$true_values),]
na_vals_single$NA_rank <- 1:nrow(na_vals_single)

## HERE - we need to subset this dataframe to include ~ 50 points, that are evenly distributed across the 1st, 2nd and 3rd third of the rank
sub_na_vals_single <- na_vals_single[dwnsmp,]

sub_na_vals_single$NA_rank <- as.factor(sub_na_vals_single$NA_rank)
na_vals_single_melt <- melt(sub_na_vals_single, by = c("NA_rank"))
na_vals_single_melt$NA_rank <- as.numeric(paste(na_vals_single_melt$NA_rank))

f1d <- ggplot(data = na_vals_single_melt, mapping = aes(x = NA_rank, y = value)) +
  geom_point() +
  geom_point(data = na_vals_single_melt[na_vals_single_melt$variable=="true_values",], colour = "red") +
  theme_bw() +
  ylim(c(min(na_vals_is_melt$value) - 1, max(na_vals_is_melt$value + 1))) +
  xlab("NA index, ranked by true value") +
  ylab("Imputed value by single imputation")

library(cowplot)

pdf("../../ICLR_paper/figures/Single_versus_multiple_imputation_NA_index_1000points_beta1_epochs250.pdf", width = 7, height = 5)
plot_grid(f1d, f1a, labels = c("a","b"))
dev.off()

pdf("../../ICLR_paper/figures/Single_versus_all_imputation_strategies_NA_index_1000points_beta1_epochs250.pdf", width = 15, height = 5)
plot_grid(f1d, f1a, f1b, f1c, nrow = 1)
dev.off()

## Export as separate plots
pdf("../../ICLR_paper/figures/Figure1A_beta1_epochs250_ranked_NA_indices.pdf", 3, 3)
print(f1d)
dev.off()

pdf("../../ICLR_paper/figures/Figure1B_beta1_epochs250_ranked_NA_indices_IS.pdf", 3, 3)
print(f1a)
dev.off()

pdf("../../ICLR_paper/figures/Figure1C_beta1_epochs250_ranked_NA_indices_MWG.pdf", 3, 3)
print(f1b)
dev.off()

pdf("../../ICLR_paper/figures/Figure1D_beta1_epochs250_ranked_NA_indices_PG.pdf", 3, 3)
print(f1c)
dev.off()


## Read in statistics from imputations
is_stats <- read.csv("importance-sampling_stats.csv", stringsAsFactors = F)
mg_stats <- read.csv("metropolis-within-gibbs_stats.csv", stringsAsFactors = F)
pg_stats <- read.csv("pseudo-gibbs_stats.csv", stringsAsFactors = F)
singimp_MAE <- 0.3024

all_stats <- merge(merge(is_stats, mg_stats), pg_stats)
all_stats_melt <- melt(all_stats, by = "imputation_strategy")
all_stats_melt$variable <- paste(all_stats_melt$variable)
all_stats_melt$value <- as.numeric(all_stats_melt$value)

# add single imputation results
all_stats_melt <- rbind(all_stats_melt, c("MAE", "single.imputation", singimp_MAE))
# add true CI values
all_stats_melt$variable <- factor(all_stats_melt$variable, levels = c("single.imputation",
                                                                      "metropolis.within.gibbs",
                                                                      "pseudo.gibbs","importance.sampling"))
all_stats_melt$value <- as.numeric(all_stats_melt$value)
all_stats_melt$label <- plyr::mapvalues(all_stats_melt$variable, c("importance.sampling","metropolis.within.gibbs",
                                                                   "pseudo.gibbs","single.imputation"),
                                        c("SIR","MWG","PG","SI"))
all_stats_melt$label <- factor(all_stats_melt$label, levels = c("SI","PG","MWG","SIR"))

pdf("../../ICLR_paper/figures/Figure1E_beta1_epochs250_95prcnt_CI_dotplot.pdf", 3, 3)
ggplot(data = all_stats_melt[all_stats_melt$imputation_strategy=="ci_95",], mapping = aes(x = label, y = value, shape = variable)) +
  geom_point() +
  geom_hline(yintercept = 0.95, linetype = "dashed") +
  theme_bw() +
  ylim(0.8,1) +
  ylab("Empirical Coverage") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = "none")
dev.off()

pdf("../../ICLR_paper/figures/Figure1F_beta1_epochs250_MAE_dotplot.pdf", 3, 3)
ggplot(data = all_stats_melt[all_stats_melt$imputation_strategy=="MAE",], mapping = aes(x = label, y = value, shape = variable)) +
  geom_point() +
  theme_bw() +
  ylim(0.28,0.33) +
  ylab("Mean Absolute Error (MAE)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = "none")
dev.off()

## Deviation at 95% confidence intervals for beta=2, epochs=250
sub_stats <- all_stats_melt[all_stats_melt$imputation_strategy=="ci_95",]
sub_stats$deviation <- sub_stats$value - 0.95

pdf("../../ICLR_paper/figures/Deviation_from_95CI_beta1_epochs250.pdf", 3, 3)
ggplot(data = sub_stats, mapping = aes(x = label, y = deviation, shape = variable)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ylab("Deviation from true 95% coverage") +
  ylim(c(-0.08,0.02)) +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x = element_blank())
dev.off()

