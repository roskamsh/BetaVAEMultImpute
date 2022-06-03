process LASSO {
    publishDir "${params.outdir}/lasso/${imputation}", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    tuple val(imputation), path(data_impute), path(data_miss), path(data_compl)

    output:
    tuple val(imputation), path("*_lasso_coeff.csv")

    script:
    """
    #!/usr/bin/env Rscript

    library(data.table)
    library(pheatmap)
    library(MASS)
    library(penalized)

    data_missing <- fread("${data_miss}",header = T)
    data_impute <- fread("${data_impute}")
    data_raw <- fread("${data_compl}",header=T)

    md <- data_raw[,1:4]
    data_raw <- as.data.frame(data_raw[,-c(1:4)])

    ## need to recompile a complete dataset with data_impute
    idx <- which(is.na(rowSums(data_missing))) # rows in the original missing dataframe that exist in data_impute
    # Essentially need to replace the rows corresponding to the index idx in data_missing with data_impute
    compl_data_impute <- data_missing
    compl_data_impute[idx,] <- data_impute
    compl_data_impute <- as.data.frame(compl_data_impute)
    matr_df <- as.matrix(compl_data_impute)

    ## configure metadata for logistic regression code
    class <- ifelse(md[["admin.disease_code"]]=="gbm", 1, 0)

    N = dim(matr_df)[1]
    D = dim(matr_df)[2]

    N0=sum(class==0)
    N1=sum(class==1)

    ## generate merged dataframe
    data = data.frame( y = class, x = matr_df)
    # optimize lambda value
    data.l1= optL1(response=as.factor(class), penalized = matr_df,  model = "logistic")
    # Determine non-zero coefficients
    data.l1.fit = penalized(response=as.factor(class), penalized = matr_df, lambda1 = data.l1[["lambda"]])
    coef(data.l1.fit )
    which(coef(data.l1.fit,"all")!=0)

    # export intercept and gene coefficients
    coeff_res <- as.data.frame(coef(data.l1.fit))

    outname <- paste0(unlist(strsplit("${data_impute}", split = "[.]"))[1], "_lasso_coeff.csv")

    write.csv(coeff_res, file = outname)
    """
}
