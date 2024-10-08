# BetaVAEMultImpute

As missing values are frequently present in genomic data, practical methods to handle missing data are necessary for downstream analyses that require complete datasets. In this work, we describe the use of a deep learning framework based on the variational autoencoder (VAE) to impute missing values using multiple imputation in transcriptomic data. 

Gene expression data is version 2 of the adjusted pan-cancer gene expression data obtained from Synapse: https://www.synapse.org/#!Synapse:syn4976369.2. Examples of preprocessing the raw data and creating missing value simulations can be found in ./preprocess.

**Build your environments**

```
conda env create --file conda_env/vae_imp_tf2.yaml
conda env create --file conda_env/lasso.yaml
```

**Set up input files**

You must have nextflow installed prior to use. 

1. Set parameters in nextflow.config and example_config_VAE.json file 

**Imputation**

This pipeline is written in nextflow to allow parallel computing across all imputation strategies.

2. Run pipeline via nextflow

```
nextflow run main.nf
```

3. View results in the output directory specified in the nextflow.config file.
