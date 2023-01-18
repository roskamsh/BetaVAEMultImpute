# BetaVAEImputation

As missing values are frequently present in genomic data, practical methods to handle missing data are necessary for downstream analyses that require complete datasets. In this work, we describe the use of a deep learning framework based on the variational autoencoder (VAE) to impute missing values in transcriptome and methylome data analysis.

The scripts contained in this repository can be used to carry out the analysis in “Genomic data imputation with variational autoencoders”. The data used in the manuscript is publicly accessible. Gene expression data is version 2 of the adjusted pan-cancer gene expression data obtained from Synapse: https://www.synapse.org/#!Synapse:syn4976369.2. DNA methylation data is the WGBS data for BLUEPRINT methylomes (2016 release) obtained from rnbeads.org: https://www.rnbeads.org/methylomes.html. Examples of preprocessing the raw data and creating missing value simulations can be found in ./preprocess.

**Build your environments**

```
conda env create --file conda_env/vae_imp_tf2.yaml
conda env create --file conda_env/lasso.yaml
```

**Set up input files**

1. Set parameters in nextflow.config and example_config_VAE.json file 

**Imputation**

This pipeline is written in nextflow to allow parallel computing across all imputation strategies.

**Compare clinical correlations**

1. Spearman correlation to histologic grade  
python cindex_spearman_cor.py

2. Cox regression coefficient to survival  
Rscript find_cox_coeff.R  
python cindex_cox_coeff.py

