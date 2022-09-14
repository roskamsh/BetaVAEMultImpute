## Evaluate LASSO regression
library(ggplot2)
library(reshape2)

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/final_analysis/beta2_epochs250/")

imps <- c("importance-sampling","metropolis-within-gibbs","pseudo-gibbs")
coeffs_list <- list()
acc_list <- list()

remove_incomplete <- FALSE

for (j in 1:length(imps)) {
  imp <- imps[j]
  
  ## Compile LASSO regression output into combined table
  files <- list.files(paste("lasso", imp, sep = "/"))
  files_coeff <- files[grep("lasso_coeff",files)]
  files_acc <- files[grep("Overall_statistics",files)]
  
  if (remove_incomplete) {
    if (imp == "importance-sampling") {
      files_coeff <- files_coeff[!(files_coeff %in% "plaus_dataset_99_lasso_coeff.csv")]
    } else if (imp == "pseudo-gibbs") {
      files_coeff <- files_coeff[!(files_coeff %in% "plaus_dataset_9_lasso_coeff.csv")]
    }
  }
  
  for (i in 1:length(files_coeff)) {
    file_coeff <- paste("lasso", imp, files_coeff[i], sep = "/")
    file_acc <- paste("lasso", imp, files_acc[i], sep = "/")
    
    df_coeff <- read.csv(file_coeff, header = T)
    df_acc <- read.csv(file_acc, header = T)
    # Rename coefficient column name by plausible dataset number
    colnames(df_coeff)[2] <- unlist(strsplit(files_coeff[i], "[.]"))[1]
    colnames(df_acc)[2] <- unlist(strsplit(files_acc[i], "[.]"))[1]
    
    if (i==1) {
      final_coeff <- df_coeff
      final_acc <- df_acc
    } else {
      final_coeff <- merge(final_coeff, df_coeff, all = TRUE)
      final_acc <- merge(final_acc, df_acc, all = TRUE)
    }
    
    if (i==length(files_coeff)) {
      final_coeff[is.na(final_coeff)] <- 0
      coeffs_list[[imp]] <- final_coeff
      
      acc_list[[imp]] <- final_acc
    }
    
  }
}

library(pheatmap)

# Read in true lasso coefficients
truevals <- read.csv("lasso/true_data_lasso_coeff.csv", header = T)
colnames(truevals)[2] <- "true_values_lasso_coeff"

# Read in single imputation coefficients
singimp <- read.csv("lasso/single-imputation/single_imputed_dataset_lasso_coeff.csv", header = T)
colnames(singimp)[2] <- "single_imputation_lasso_coeff"

## Heatmap for Metropolis-within-Gibbs
coeff_is <- coeffs_list$`importance-sampling`
coeff_mg <- coeffs_list$`metropolis-within-gibbs`
coeff_pg <- coeffs_list$`pseudo-gibbs`

# Add in truevals
coeff_is <- merge(coeff_is, truevals, all = T)
coeff_is <- merge(coeff_is, singimp, all = T)
coeff_is[is.na(coeff_is)] <- 0

##############################
## Importance sampling ##
##############################
rownames(coeff_is) <- coeff_is$X
coeff_is$X <- NULL

is_hm <- pheatmap(coeff_is, show_rownames = T, scale = "row", 
                  color = colorRampPalette(c("navy", "white", "firebrick3"))(50))

## What are the summary stats of the coefficients?
# Mean of imputed coefficients
coeff_is$mean_imputed_lasso_coeff <- rowMeans(coeff_is[,1:100])

# What if you did a scatterplot of ranked coefficients and plotted the true values overtop?
coeff_is <- coeff_is[order(coeff_is$true_values_lasso_coeff),]
coeff_is$true_values_lasso_coeff <- as.factor(coeff_is$true_values_lasso_coeff)
coeff_is$single_imputation_lasso_coeff <- as.factor(coeff_is$single_imputation_lasso_coeff)
coeff_is$mean_imputed_lasso_coeff <- as.factor(coeff_is$mean_imputed_lasso_coeff)
coeff_is_melt <- melt(coeff_is, by = c("true_values_lasso_coeff", "single_imputation_lasso_coeff",
                                       "mean_imputed_lasso_coeff"))

## What about numeric?
coeff_is_melt$true_values_lasso_coeff <- as.numeric(paste(coeff_is_melt$true_values_lasso_coeff))
coeff_is_melt$single_imputation_lasso_coeff <- as.numeric(paste(coeff_is_melt$single_imputation_lasso_coeff))
coeff_is_melt$mean_imputed_lasso_coeff <- as.numeric(paste(coeff_is_melt$mean_imputed_lasso_coeff))

ggplot(data = coeff_is_melt, mapping = aes(x = true_values_lasso_coeff, y = value)) +
  geom_point(alpha = 0.5) +
  geom_point(aes(x = true_values_lasso_coeff, y = single_imputation_lasso_coeff), colour = "blue") +
  #geom_point(aes(x = true_values_lasso_coeff, y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(x = true_values_lasso_coeff, y = true_values_lasso_coeff), colour = "red") +
  xlab("LASSO coefficients on true data matrix (red)") +
  ylab("Imputed value across m=100 \ndatasets (importance sampling)")

#### Plot mean and single imputation on here as well
if (remove_incomplete) {
  coeff_is_melt$gene <- rep(rownames(coeff_is), 99)
} else {
  coeff_is_melt$gene <- rep(rownames(coeff_is), 100)
}

ggplot(data = coeff_is_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  #geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90))

# Remove intercept from plot to show spread better
p1 <- ggplot(data = coeff_is_melt[coeff_is_melt$gene != "(Intercept)",], mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  theme_bw() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  #geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90))

# Deviation of mean imputed lasso coefficients compared to true values, plot this deviation against true value coefficient
coeff_is_melt$dev_coeff <- coeff_is_melt$true_values_lasso_coeff - coeff_is_melt$mean_imputed_lasso_coeff

ggplot(data = coeff_is_melt, mapping = aes(x = dev_coeff, y = true_values_lasso_coeff)) +
  geom_point() +
  theme_bw()

## What is happening with these outlier points for the intercept??
# Extract which plausible datasets these are from
outlier_dats_is <- coeff_is_melt[coeff_is_melt$gene=="(Intercept)" & coeff_is_melt$value > -20,]$variable

# Write table
write.csv(as.data.frame(outlier_dats_is), "lasso/Intercept_outlier_plaus_datasets_importance-sampling.csv")

# What do these "outlier datasets" look like int he boxplot of lasso coefficients if we facet out by this column?
coeff_is_melt$is_intercept_outlier <- ifelse(coeff_is_melt$variable %in% paste(outlier_dats_is), TRUE, FALSE)

ggplot(data = coeff_is_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue", size = 0.5) +
  geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow", size = 0.5) +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red", size = 0.5) +
  facet_wrap(~ is_intercept_outlier, nrow = 2) +
  theme(axis.text.x = element_text(angle = 90))


## What about 95% CIs for LASSO regression coefficients?
# Compute mean and standard deviation across all coefficients
if (remove_incomplete) {
  coeff_is$stdev_imputed_lasso_coeff <- apply(coeff_is[,1:99],1,sd)
} else {
  coeff_is$stdev_imputed_lasso_coeff <- apply(coeff_is[,1:100],1,sd)
}


# Compute 95% confidence intervals for each NA index
CIs <- sapply(1:nrow(coeff_is), function(x) {
  
  if (remove_incomplete) {
    n <- 99
  } else {
    n <- 100
  }
  
  subdat <- coeff_is[x,]
  s <- subdat$stdev_imputed_lasso_coeff
  mn <- as.numeric(paste(subdat$mean_imputed_lasso_coeff))
  grtruth <- as.numeric(paste(subdat$true_values_lasso_coeff))
  
  margin <- qnorm(0.975)*s
  upper_interval <- mn + margin
  lower_interval <- mn - margin
  
  is_95 <- grtruth < upper_interval & grtruth > lower_interval
  
  final_res <- c(lower_interval,upper_interval,is_95)
})

## Now add this into our table na_vals
coeff_is <- data.frame(coeff_is, as.data.frame(t(CIs)))
colnames(coeff_is)[(ncol(coeff_is)-2):ncol(coeff_is)] <- c("lower_interval","upper_interval","in_95_CI")
coeff_is$in_95_CI <- as.logical(coeff_is$in_95_CI)

# What does this look like?
summary(coeff_is$in_95_CI) 
sum(coeff_is$in_95_CI==TRUE) / nrow(coeff_is) # 96.6% 

# MAE on single imputed value and mean of imputed values for lasso coefficients
MAE <- mean(abs(as.numeric(paste(coeff_is$true_values_lasso_coeff)) - as.numeric(paste(coeff_is$mean_imputed_lasso_coeff))))
# 0.069

# number of coefficients where zero is not contained in the 95% CI
coeff_is$nonzero_95CI <- ifelse((coeff_is$lower_interval > 0 & coeff_is$upper_interval > 0) | 
                                  (coeff_is$lower_interval < 0 & coeff_is$upper_interval < 0), TRUE, FALSE)

# What does this look like?
summary(coeff_is$nonzero_95CI) 
sum(coeff_is$nonzero_95CI==TRUE) / nrow(coeff_is) # 8% 

## Inclusion probability
# Compute the fraction of datasets that each gene (across all genes that are non-zero) is non-zero
coeff_is$incl_prob <- rowSums(coeff_is[,1:100]!=0)/ncol(coeff_is[,1:100])

hist(coeff_is$incl_prob)
sum(coeff_is$incl_prob>0.5) # 25 genes have > 0.5 inclusion probability
incl_genes_is <- rownames(coeff_is[coeff_is$incl_prob>0.5,])

incl_genes_is %in% truevals$X # 2 false positive

fpos <- incl_genes_is[!(incl_genes_is %in% truevals$X)]
# What are the inclusion probabilities for these?
coeff_is[rownames(coeff_is) %in% fpos,]$incl_prob

# Look at dotplot of inclusion probabilities with genes on x axis and incl prob on y axis, and colour by whether in true or not
coeff_is$gene <- rownames(coeff_is)
rank <- rownames(coeff_is[order(coeff_is$incl_prob),]) # Rank gene column by inclusion probability
coeff_is$gene <- factor(coeff_is$gene, levels = rank)
coeff_is$is_in_true_coeff <- ifelse(coeff_is$gene %in% truevals$X, TRUE, FALSE)

pdf("../../ICLR_paper/figures/inclusion_probability_ranked_plot_importance-sampling.pdf", height = 5, width = 10)
ggplot(data = coeff_is, mapping = aes(x = gene, y = incl_prob, colour = is_in_true_coeff)) +
  geom_point() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 4))
dev.off()

## What does the boxplot of lasso coefficients look like when ranked by inclusion probability?
rank <- rownames(coeff_is[order(coeff_is$incl_prob, decreasing = TRUE),])
coeff_is_melt$gene <- factor(coeff_is_melt$gene, levels = rank)

p1rank <- ggplot(data = coeff_is_melt[coeff_is_melt$gene != "(Intercept)",], mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  ylab("LASSO Coefficient") +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90, size = 4))

pdf("../../ICLR_paper/figures/LASSO_coefficients_heatmap_importance_sampling.pdf", width = 10, height = 5)
print(p1rank)
dev.off()


##############################
#### METROPOLIS-WITHIN-GIBBS ####
##############################

# Add in truevals
coeff_mg <- merge(coeff_mg, truevals, all = T)
coeff_mg <- merge(coeff_mg, singimp, all = T)
coeff_mg[is.na(coeff_mg)] <- 0

rownames(coeff_mg) <- coeff_mg$X
coeff_mg$X <- NULL

mg_hm <- pheatmap(coeff_mg, show_rownames = T, scale = "row", 
                  color = colorRampPalette(c("navy", "white", "firebrick3"))(50))

## What are the summary stats of the coefficients?
# Mean of imputed coefficients
coeff_mg$mean_imputed_lasso_coeff <- rowMeans(coeff_mg[,1:100])

# What if you did a scatterplot of ranked coefficients and plotted the true values overtop?
coeff_mg <- coeff_mg[order(coeff_mg$true_values_lasso_coeff),]
coeff_mg$true_values_lasso_coeff <- as.factor(coeff_mg$true_values_lasso_coeff)
coeff_mg$single_imputation_lasso_coeff <- as.factor(coeff_mg$single_imputation_lasso_coeff)
coeff_mg$mean_imputed_lasso_coeff <- as.factor(coeff_mg$mean_imputed_lasso_coeff)
coeff_mg_melt <- melt(coeff_mg, by = c("true_values_lasso_coeff", "single_imputation_lasso_coeff",
                                       "mean_imputed_lasso_coeff"))

## What about numeric?
coeff_mg_melt$true_values_lasso_coeff <- as.numeric(paste(coeff_mg_melt$true_values_lasso_coeff))
coeff_mg_melt$single_imputation_lasso_coeff <- as.numeric(paste(coeff_mg_melt$single_imputation_lasso_coeff))
coeff_mg_melt$mean_imputed_lasso_coeff <- as.numeric(paste(coeff_mg_melt$mean_imputed_lasso_coeff))

ggplot(data = coeff_mg_melt, mapping = aes(x = true_values_lasso_coeff, y = value)) +
  geom_point(alpha = 0.5) +
  geom_point(aes(x = true_values_lasso_coeff, y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(x = true_values_lasso_coeff, y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(x = true_values_lasso_coeff, y = true_values_lasso_coeff), colour = "red") +
  xlab("LASSO coefficients on true data matrix (red)") +
  ylab("Imputed value across m=100 \ndatasets (metreopolis-within-Gibbs)")

#### Plot mean and single imputation on here as well
coeff_mg_melt$gene <- rep(rownames(coeff_mg), 100)

ggplot(data = coeff_mg_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90))

# Deviation of mean imputed lasso coefficients compared to true values, plot this deviation against true value coefficient
coeff_mg_melt$dev_coeff <- coeff_mg_melt$true_values_lasso_coeff - coeff_mg_melt$mean_imputed_lasso_coeff

ggplot(data = coeff_mg_melt, mapping = aes(x = dev_coeff, y = true_values_lasso_coeff)) +
  geom_point() +
  theme_bw()

### ASSESS HERE
## What is happening with these outlier points for the intercept??
# Extract which plausible datasets these are from
outlier_dats_mg <- coeff_mg_melt[coeff_mg_melt$gene=="(Intercept)" & coeff_mg_melt$value > -20,]$variable

# Write table
write.csv(as.data.frame(outlier_dats_mg), "lasso/Intercept_outlier_plaus_datasets_metropolis-within-gibbs.csv")

# What do these "outlier datasets" look like int he boxplot of lasso coefficients if we facet out by this column?
coeff_mg_melt$is_intercept_outlier <- ifelse(coeff_mg_melt$variable %in% paste(outlier_dats_mg), TRUE, FALSE)

ggplot(data = coeff_mg_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue", size = 0.5) +
  geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow", size = 0.5) +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red", size = 0.5) +
  facet_wrap(~ is_intercept_outlier, nrow = 2) +
  theme(axis.text.x = element_text(angle = 90))

## What about 95% CIs for LASSO regression coefficients?
# Compute mean and standard deviation across all coefficients
coeff_mg$stdev_imputed_lasso_coeff <- apply(coeff_mg[,1:100],1,sd)

# Compute 95% confidence intervals for each NA index
CIs <- sapply(1:nrow(coeff_mg), function(x) {
  n <- 100
  subdat <- coeff_mg[x,]
  s <- subdat$stdev_imputed_lasso_coeff
  mn <- as.numeric(paste(subdat$mean_imputed_lasso_coeff))
  grtruth <- as.numeric(paste(subdat$true_values_lasso_coeff))
  
  margin <- qnorm(0.975)*s
  upper_interval <- mn + margin
  lower_interval <- mn - margin
  
  is_95 <- grtruth < upper_interval & grtruth > lower_interval
  
  final_res <- c(lower_interval,upper_interval,is_95)
})

## Now add this into our table na_vals
coeff_mg <- data.frame(coeff_mg, as.data.frame(t(CIs)))
colnames(coeff_mg)[(ncol(coeff_mg)-2):ncol(coeff_mg)] <- c("lower_interval","upper_interval","in_95_CI")
coeff_mg$in_95_CI <- as.logical(coeff_mg$in_95_CI)

# What does this look like?
summary(coeff_mg$in_95_CI) ## Fixed now!!
sum(coeff_mg$in_95_CI==TRUE) / nrow(coeff_mg) # 98.6% 

# MAE on single imputed value and mean of imputed values for lasso coefficients
MAE <- mean(abs(as.numeric(paste(coeff_mg$true_values_lasso_coeff)) - as.numeric(paste(coeff_mg$mean_imputed_lasso_coeff))))
# 0.1536

# number of coefficients where zero is not contained in the 95% CI
coeff_mg$nonzero_95CI <- ifelse((coeff_mg$lower_interval > 0 & coeff_mg$upper_interval > 0) | 
                                  (coeff_mg$lower_interval < 0 & coeff_mg$upper_interval < 0), TRUE, FALSE)

# What does this look like?
summary(coeff_mg$nonzero_95CI) ## Fixed now!!
sum(coeff_mg$nonzero_95CI==TRUE) / nrow(coeff_mg) 

## Inclusion probability
# Compute the fraction of datasets that each gene (across all genes that are non-zero) is non-zero
coeff_mg$incl_prob <- rowSums(coeff_mg[,1:100]!=0)/ncol(coeff_mg[,1:100])

hist(coeff_mg$incl_prob)
sum(coeff_mg$incl_prob>0.5) # 25 genes have > 0.5 inclusion probability
incl_genes_mg <- rownames(coeff_mg[coeff_mg$incl_prob>0.5,])

incl_genes_mg %in% truevals$X # 2 false positive

fpos <- incl_genes_mg[!(incl_genes_mg %in% truevals$X)]
# What are the inclusion probabilities for these?
coeff_mg[rownames(coeff_mg) %in% fpos,]$incl_prob

# Look at dotplot of inclusion probabilities with genes on x axis and incl prob on y axis, and colour by whether in true or not
coeff_mg$gene <- rownames(coeff_mg)
rank <- rownames(coeff_mg[order(coeff_mg$incl_prob),]) # Rank gene column by inclusion probability
coeff_mg$gene <- factor(coeff_mg$gene, levels = rank)
coeff_mg$is_in_true_coeff <- ifelse(coeff_mg$gene %in% truevals$X, TRUE, FALSE)

pdf("../../ICLR_paper/figures/inclusion_probability_ranked_plot_metropolis-within-gibbs.pdf", height = 5, width = 10)
ggplot(data = coeff_mg, mapping = aes(x = gene, y = incl_prob, colour = is_in_true_coeff)) +
  geom_point() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 4))
dev.off()

## What does the boxplot of lasso coefficients look like when ranked by inclusion probability?
rank <- rownames(coeff_mg[order(coeff_mg$incl_prob, decreasing = TRUE),])
coeff_mg_melt$gene <- factor(coeff_mg_melt$gene, levels = rank)

p2rank <- ggplot(data = coeff_mg_melt[coeff_mg_melt$gene != "(Intercept)",], mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  ylab("LASSO Coefficient") +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90,size = 4))

pdf("../../ICLR_paper/figures/LASSO_coefficients_heatmap_metropolis-within-gibbs.pdf", width = 10, height = 5)
print(p2rank)
dev.off()


#########################
# PSEUDO-GIBBS
#########################
# Add in truevals
coeff_pg <- merge(coeff_pg, truevals, all = T)
coeff_pg <- merge(coeff_pg, singimp, all = T)
coeff_pg[is.na(coeff_pg)] <- 0

rownames(coeff_pg) <- coeff_pg$X
coeff_pg$X <- NULL

pg_hm <- pheatmap(coeff_pg, show_rownames = T, scale = "row", 
                  color = colorRampPalette(c("navy", "white", "firebrick3"))(50))

## What are the summary stats of the coefficients?
# Mean of imputed coefficients
if (remove_incomplete) {
  coeff_pg$mean_imputed_lasso_coeff <- rowMeans(coeff_pg[,1:99])
} else {
  coeff_pg$mean_imputed_lasso_coeff <- rowMeans(coeff_pg[,1:100])
}


# What if you did a scatterplot of ranked coefficients and plotted the true values overtop?
coeff_pg <- coeff_pg[order(coeff_pg$true_values_lasso_coeff),]
coeff_pg$true_values_lasso_coeff <- as.factor(coeff_pg$true_values_lasso_coeff)
coeff_pg$single_imputation_lasso_coeff <- as.factor(coeff_pg$single_imputation_lasso_coeff)
coeff_pg$mean_imputed_lasso_coeff <- as.factor(coeff_pg$mean_imputed_lasso_coeff)
coeff_pg_melt <- melt(coeff_pg, by = c("true_values_lasso_coeff", "single_imputation_lasso_coeff",
                                       "mean_imputed_lasso_coeff"))

## What about numeric?
coeff_pg_melt$true_values_lasso_coeff <- as.numeric(paste(coeff_pg_melt$true_values_lasso_coeff))
coeff_pg_melt$single_imputation_lasso_coeff <- as.numeric(paste(coeff_pg_melt$single_imputation_lasso_coeff))
coeff_pg_melt$mean_imputed_lasso_coeff <- as.numeric(paste(coeff_pg_melt$mean_imputed_lasso_coeff))

ggplot(data = coeff_pg_melt, mapping = aes(x = true_values_lasso_coeff, y = value)) +
  geom_point(alpha = 0.5) +
  geom_point(aes(x = true_values_lasso_coeff, y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(x = true_values_lasso_coeff, y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(x = true_values_lasso_coeff, y = true_values_lasso_coeff), colour = "red") +
  xlab("LASSO coefficients on true data matrix (red)") +
  ylab("Imputed value across m=100 \ndatasets (pseudo-gibbs)")

#### Plot mean and single imputation on here as well
if (remove_incomplete) {
  coeff_pg_melt$gene <- rep(rownames(coeff_pg), 99)
} else {
  coeff_pg_melt$gene <- rep(rownames(coeff_pg), 100)
}


ggplot(data = coeff_pg_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90))

# Deviation of mean imputed lasso coefficients compared to true values, plot this deviation against true value coefficient
coeff_pg_melt$dev_coeff <- coeff_pg_melt$true_values_lasso_coeff - coeff_pg_melt$mean_imputed_lasso_coeff

ggplot(data = coeff_pg_melt, mapping = aes(x = dev_coeff, y = true_values_lasso_coeff)) +
  geom_point() +
  theme_bw()

### ASSESS HERE
## What is happening with these outlier points for the intercept??
# Extract which plausible datasets these are from
outlier_dats_pg <- coeff_pg_melt[coeff_pg_melt$gene=="(Intercept)" & coeff_pg_melt$value > -20,]$variable

# Write table
write.csv(as.data.frame(outlier_dats_pg), "lasso/Intercept_outlier_plaus_datasets_pseudo-gibbs.csv")

# What do these "outlier datasets" look like int he boxplot of lasso coefficients if we facet out by this column?
coeff_pg_melt$is_intercept_outlier <- ifelse(coeff_pg_melt$variable %in% paste(outlier_dats_pg), TRUE, FALSE)

ggplot(data = coeff_pg_melt, mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue", size = 0.5) +
  geom_point(aes(y = mean_imputed_lasso_coeff), colour = "yellow", size = 0.5) +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red", size = 0.5) +
  facet_wrap(~ is_intercept_outlier, nrow = 2) +
  theme(axis.text.x = element_text(angle = 90))

## What about 95% CIs for LASSO regression coefficients?
# Compute mean and standard deviation across all coefficients
if (remove_incomplete) {
  coeff_pg$stdev_imputed_lasso_coeff <- apply(coeff_pg[,1:99],1,sd)
} else {
  coeff_pg$stdev_imputed_lasso_coeff <- apply(coeff_pg[,1:100],1,sd)
}


# Compute 95% confidence intervals for each NA index
CIs <- sapply(1:nrow(coeff_pg), function(x) {
  if (remove_incomplete) {
    n <- 99
  } else {
    n <- 100
  }
  subdat <- coeff_pg[x,]
  s <- subdat$stdev_imputed_lasso_coeff
  mn <- as.numeric(paste(subdat$mean_imputed_lasso_coeff))
  grtruth <- as.numeric(paste(subdat$true_values_lasso_coeff))
  
  margin <- qnorm(0.975)*s
  upper_interval <- mn + margin
  lower_interval <- mn - margin
  
  is_95 <- grtruth < upper_interval & grtruth > lower_interval
  
  final_res <- c(lower_interval,upper_interval,is_95)
})

## Now add this into our table na_vals
coeff_pg <- data.frame(coeff_pg, as.data.frame(t(CIs)))
colnames(coeff_pg)[(ncol(coeff_pg)-2):ncol(coeff_pg)] <- c("lower_interval","upper_interval","in_95_CI")
coeff_pg$in_95_CI <- as.logical(coeff_pg$in_95_CI)

# What does this look like?
summary(coeff_pg$in_95_CI) ## Fixed now!!
sum(coeff_pg$in_95_CI==TRUE) / nrow(coeff_pg) # 98.6% 

# Which gene is FALSE?
rownames(coeff_pg[coeff_pg$in_95_CI==FALSE,])

# MAE on single imputed value and mean of imputed values for lasso coefficients
MAE <- mean(abs(as.numeric(paste(coeff_pg$true_values_lasso_coeff)) - as.numeric(paste(coeff_pg$mean_imputed_lasso_coeff))))

# number of coefficients where zero is not contained in the 95% CI
coeff_pg$nonzero_95CI <- ifelse((coeff_pg$lower_interval > 0 & coeff_pg$upper_interval > 0) | 
                                  (coeff_pg$lower_interval < 0 & coeff_pg$upper_interval < 0), TRUE, FALSE)

# What does this look like?
summary(coeff_pg$nonzero_95CI) ## Fixed now!!
sum(coeff_pg$nonzero_95CI==TRUE) / nrow(coeff_pg) # 98.6% 

## Inclusion probability
# Compute the fraction of datasets that each gene (across all genes that are non-zero) is non-zero
coeff_pg$incl_prob <- rowSums(coeff_pg[,1:100]!=0)/ncol(coeff_pg[,1:100])

hist(coeff_pg$incl_prob)
sum(coeff_pg$incl_prob>0.5) # 25 genes have > 0.5 inclusion probability
incl_genes_pg <- rownames(coeff_pg[coeff_pg$incl_prob>0.5,])

incl_genes_pg %in% truevals$X # 2 false positive

fpos <- incl_genes_pg[!(incl_genes_pg %in% truevals$X)]
# What are the inclusion probabilities for these?
coeff_pg[rownames(coeff_pg) %in% fpos,]$incl_prob

# Look at dotplot of inclusion probabilities with genes on x axis and incl prob on y axis, and colour by whether in true or not
coeff_pg$gene <- rownames(coeff_pg)
rank <- rownames(coeff_pg[order(coeff_pg$incl_prob),]) # Rank gene column by inclusion probability
coeff_pg$gene <- factor(coeff_pg$gene, levels = rank)
coeff_pg$is_in_true_coeff <- ifelse(coeff_pg$gene %in% truevals$X, TRUE, FALSE)

pdf("../../ICLR_paper/figures/inclusion_probability_ranked_plot_pseudo-gibbs.pdf", height = 5, width = 10)
ggplot(data = coeff_pg, mapping = aes(x = gene, y = incl_prob, colour = is_in_true_coeff)) +
  geom_point() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 4))
dev.off()

## What does the boxplot of lasso coefficients look like when ranked by inclusion probability?
rank <- rownames(coeff_pg[order(coeff_pg$incl_prob, decreasing = TRUE),])
coeff_pg_melt$gene <- factor(coeff_pg_melt$gene, levels = rank)

p3rank <- ggplot(data = coeff_pg_melt[coeff_pg_melt$gene != "(Intercept)",], mapping = aes(x = gene, y = value)) +
  geom_boxplot() +
  ylab("LASSO Coefficient") +
  geom_point(aes(y = single_imputation_lasso_coeff), colour = "blue") +
  geom_point(aes(y = true_values_lasso_coeff), colour = "red") +
  theme(axis.text.x = element_text(angle = 90, size = 4))

pdf("../../ICLR_paper/figures/LASSO_coefficients_heatmap_pseudo-gibbs.pdf", width = 10, height = 5)
print(p3rank)
dev.off()


##########################################
## Further analysis comparing all three imputations together
##########################################

# Single imputation
singimp_MAE <- mean(abs(as.numeric(paste(coeff_pg$true_values_lasso_coeff)) - as.numeric(paste(coeff_pg$single_imputation_lasso_coeff))))

## What about the datasets that are "outliers" in all 3 cases?
mg_pg <- intersect(paste(outlier_dats_pg), paste(outlier_dats_mg))
is_pg_mg <- intersect(mg_pg, paste(outlier_dats_is))

## How many non-zero coefficients for single imputation
dim(singimp)

## Look at the nonzero95CI in the true values?
table(rownames(coeff_is[coeff_is$nonzero_95CI,]) %in% truevals$X)
table(rownames(coeff_mg[coeff_mg$nonzero_95CI,]) %in% truevals$X)
table(rownames(coeff_pg[coeff_pg$nonzero_95CI,]) %in% truevals$X)

table(truevals$X %in% singimp$X)

length(intersect(rownames(coeff_is[coeff_mg$nonzero_95CI,]), rownames(coeff_pg[coeff_pg$nonzero_95CI,])))

length(intersect(rownames(coeff_mg[coeff_mg$nonzero_95CI,]), singimp$X))

# Amazing! All of the significant genes from the multiple imputation are contained within the true data coefficient sets
# Also I think that the importance sampling actually picks up one coefficient that is missed in the single imputation

# Let's make an UpSet plot looking at the overlap of sets between:
# 1. true lasso coefficients
# 2. Single imp lasso coefficients
# 3. nonzero 95%CI for all three imp strategies
# 4. inclusion probability of 0.5 for all three imp strategies
library(UpSetR)

gene_sets <- list(GT = truevals$X, SI = singimp$X, 
                  SIR_nonzero_95CI = rownames(coeff_is[coeff_is$nonzero_95CI,]), 
                  MWG_nonzero_95CI = rownames(coeff_mg[coeff_mg$nonzero_95CI,]), 
                  PG_nonzero_95CI = rownames(coeff_pg[coeff_pg$nonzero_95CI,]),
                  SIR_P_incl_0.5 = incl_genes_is, 
                  MWG_P_incl_0.5 = incl_genes_mg, 
                  PG_P_incl_0.5 = incl_genes_pg)

pdf("../../ICLR_paper/figures/Upset_plot_LASSO_coefficients.pdf", 7, 4, useDingbats = T)
upset(fromList(gene_sets), order.by = c("freq", "degree"), decreasing = c(TRUE,FALSE), nsets = 8)
dev.off()

####### CLASSIFICATION ACCURACY ############
acc_is <- acc_list$`importance-sampling`
acc_is_melt <- melt(acc_is, by = X)
acc_is_melt$dataset <- gsub("_Overall_statistics","",acc_is_melt$variable)
acc_is_melt$Imputation_strategy <- "importance-sampling"

acc_mg <- acc_list$`metropolis-within-gibbs`
acc_mg_melt <- melt(acc_mg, by = X)
acc_mg_melt$dataset <- gsub("_Overall_statistics","",acc_mg_melt$variable)
acc_mg_melt$Imputation_strategy <- "metropolis-within-gibbs"

acc_pg <- acc_list$`pseudo-gibbs`
acc_pg_melt <- melt(acc_pg, by = X)
acc_pg_melt$dataset <- gsub("_Overall_statistics","",acc_pg_melt$variable)
acc_pg_melt$Imputation_strategy <- "pseudo-gibbs"

acc_singimp <- read.csv("lasso/single-imputation/single_imputed_dataset_Overall_statistics.csv")
singimp_acc_val <- acc_singimp[1,2]

acc_truevals <- read.csv("lasso/true_values_Overall_statistics.csv")
truevals_acc_val <- acc_truevals[1,2]

acc_all <- rbind(acc_pg_melt, rbind(acc_is_melt, acc_mg_melt))
acc_all$singimp_accuracy <- singimp_acc_val
acc_all$truevals_accuracy <- truevals_acc_val

ggplot(data = acc_all[acc_all$X=="Accuracy",], mapping = (aes(x = Imputation_strategy, y = value))) +
  geom_boxplot() +
  geom_point(aes(y = singimp_accuracy), colour = "yellow") +
  geom_point(aes(y = truevals_accuracy), colour = "red")


