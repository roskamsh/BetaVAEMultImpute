process COMPILE_NA_INDICES {
    publishDir "${params.outdir}/multiple_imputation/${imputation}", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    tuple val(imputation), path(na_indices)

    output:
    tuple val(imputation), path("${imputation}_compiled_NA_indices.csv")

    script:
    """
    #!/usr/bin/env Rscript

    print("${imputation}")
    files <- list.files()
    files <- files[grep("NA",files)]
    files <- files[!(files %in% "compiled_NA_indices.csv")]

    for (i in 1:length(files)) {
        df <- read.csv(files[i], row.names = 1, stringsAsFactors = F)
        print(paste("reading in file number", i))
        if (i == 1) {
            final <- df
        } else {
            datname <- colnames(df)[2]
            final <- data.frame(final, df[,2])
            colnames(final)[i+1] <- datname
        }
    }

    outname = paste0("${imputation}", '_compiled_NA_indices.csv')
    write.csv(final, outname, row.names = F)
    """
}

process COMPUTE_CIs {
    publishDir "${params.outdir}/multiple_imputation", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    tuple val(imputation), path(na_indices)

    output:
    path("${imputation}_stats.csv")

    script:
    """
    #!/usr/bin/env python

    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import json

    from sklearn.preprocessing import StandardScaler

    res = pd.read_csv("${na_indices}").values

    # Assign first column of values to new variable and then remove it from res
    truevals = res[:,0]
    impvals = res[:,1:]

    means = np.mean(impvals, axis=1)
    st_devs = np.std(impvals, axis=1)
    differences = np.abs(truevals - means)
    n_deviations = differences / st_devs
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    print('prop 90:', prop_90)
    print('prop 95:', prop_95)
    print('prop 99:', prop_99)

    differences = np.abs(truevals - means)
    mae = np.mean(differences)
    print('average absolute error:', mae)

    res = ["${imputation}", mae, prop_90, prop_95, prop_99]

    # Make pandas dataframe
    out_table = pd.DataFrame(res, index = ["imputation_strategy","MAE","ci_90","ci_95","ci_99"])

    # export table
    out_table.to_csv("${imputation}_stats.csv", header = False)
    """
}

