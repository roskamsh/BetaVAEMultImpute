process COMPILE_NA_INDICES {
    publishDir "${params.outdir}/multiple-imputation/${imputation}/beta_${beta}", mode: "copy"
    cpus 1
    memory '5 GB'

    input:
    tuple val(beta), val(imputation), path(na_indices)

    output:
    tuple val(beta), val(imputation), path("${imputation}_beta${beta}_compiled_NA_indices.csv")

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

    outname = paste0("${imputation}", "_beta", "${beta}",'_compiled_NA_indices.csv')
    write.csv(final, outname, row.names = F)
    """
}

process COMPUTE_CIs {
    publishDir "${params.outdir}/multiple-imputation/beta_${beta}", mode: "copy"
    cpus 1
    memory '5 GB'

    input:
    tuple val(beta), val(imputation), path(na_indices)

    output:
    tuple val(beta), path("${imputation}_beta${beta}_stats.csv")

    script:
    """
    #!/usr/bin/env python

    import numpy as np
    import pandas as pd

    fname = "${na_indices}"
    beta = "${beta}"
    imputation = "${imputation}"

    res = pd.read_csv(fname).values

    # Assign first column of values to new variable and then remove it from res
    truevals = res[:,0]
    impvals = res[:,1:]

    # Compute statistics across all M datasets
    means = np.mean(impvals, axis=1)
    st_devs = np.std(impvals, axis=1)
    differences = np.abs(truevals - means)
    n_deviations = differences / st_devs
    mae = np.mean(differences)
    print('average absolute error:', mae)

    cis = [70,75,80,85,90,95,99]
    alphas = [1.036,1.150,1.282,1.440,1.645,1.960,2.576]

    ecs = []
    tradeoff_mae_ecs = []
    for i,alpha in enumerate(alphas):
        ci_level = cis[i]
        ec = sum(n_deviations < alpha) / len(n_deviations)
        tradeoff_mae_ec = mae + 0.5*max(0, ci_level/100 - ec)
        ecs.append(ec)
        tradeoff_mae_ecs.append(tradeoff_mae_ec)

    ec_strings = ["EC_" + str(ci) for ci in cis]
    tradeoff_strings = ["tradeoff_mae_ec_" + str(ci) for ci in cis]
    res = [beta, imputation, mae] + ecs + tradeoff_mae_ecs

    # Make pandas dataframe
    out_table = pd.DataFrame(res, index = ["beta","imputation_strategy","MAE"] + ec_strings + tradeoff_strings)

    # export table
    out_table.to_csv(f"{imputation}_beta{beta}_stats.csv", header = False)
    """
}

process COMPUTE_PERCENTILES {
    publishDir "${params.outdir}/multiple-imputation/beta_${beta}", mode: "copy"
    cpus 1
    memory '5 GB'

    input:
    tuple val(beta), val(imputation), path(na_indices)

    output:
    tuple val(beta), path("${imputation}_beta${beta}_imputation_percentiles.csv")

    script:
    """
    #!/usr/bin/env python

    import numpy as np
    import pandas as pd  

    res = pd.read_csv("${na_indices}").values

    # Assign first column of values to new variable and then remove it from res
    truevals = res[:,0]
    impvals = res[:,1:]

    prcntiles = [25, 50, 75, 95, 99]
    prcntiles_names = ["CI_" + str(s) for s in prcntiles]
    out = np.zeros((1, len(prcntiles)))

    for i in range(len(prcntiles)):
        # determine lower and upper bounds based on confidence interval coverage
        lowval = 50 - (prcntiles[i]/2)
        highval = 50 + (prcntiles[i]/2)

        # compute percentiles of imputed values based on low and high bounds
        lower = np.percentile(impvals, lowval, axis = 1) 
        higher = np.percentile(impvals, highval, axis = 1)

        # What percentage of values are in 
        is_CI = (truevals > lower) & (truevals < higher)
        prcnt = sum(is_CI)/len(is_CI)

        print("${imputation}: Coverage of CI", prcntiles[i], ":", prcnt)

        out[0][i] = prcnt

    out_table = pd.DataFrame(out, index = ["${imputation}"], columns = prcntiles_names)
    out_table["beta"] = "${beta}"

    # export table
    out_table.to_csv("${imputation}_beta${beta}_imputation_percentiles.csv", header = True,index=False) 
    """
}


process SUMMARISE_APPROX_LOGLIK {
    publishDir "${params.outdir}/multiple-imputation/beta_${beta}", mode: "copy"
    cpus 1
    memory '5 GB' 

    input:
    tuple val(beta), val(imputation), path(loglik)

    output:
    tuple val(beta), path("Approx_loglik_beta${beta}_summarized.csv")

    script:
    """
    #!/usr/bin/env python

    import pandas as pd
    import numpy as np

    fname = "${loglik}"
    beta = "${beta}" 

    res = pd.read_csv(fname)

    median_p = np.median(res[["mcmc_p"]].values)
    median_q = np.median(res[["mcmc_q"]].values)

    bound_sup_p = np.quantile(res[["mcmc_p"]].values, 0.75)
    bound_sup_q = np.quantile(res[["mcmc_q"]].values, 0.75) 
    bound_inf_p = np.quantile(res[["mcmc_p"]].values, 0.25)
    bound_inf_q = np.quantile(res[["mcmc_q"]].values, 0.25) 

    df = pd.DataFrame({'beta': [beta,beta], 'median_approxloglik': [median_p,median_q], 
                    'bound_inf': [bound_inf_p,bound_inf_q], 'bound_sup': [bound_sup_p,bound_sup_q],
                    'wrt': ["p","q"]})

    df.to_csv("Approx_loglik_beta${beta}_summarized.csv",index=False)
    """
}


process COMPUTE_MAE_SINGLE {
    publishDir "${imputation == 'mean-imputation' ? params.outdir + '/mean-imputation' : params.outdir + '/single-imputation/beta_' + beta}", mode: "copy"
    cpus 1
    memory '5 GB'

    input:
    tuple val(beta), val(imputation), path(na_indices)

    output:
    tuple val(beta), path("${imputation}_beta${beta}_stats.csv")

    script:
    """
    #!/usr/bin/env python

    import numpy as np
    import pandas as pd

    res = pd.read_csv("${na_indices}").values

    # Assign first column of values to new variable and then remove it from res
    truevals = res[:,1]
    impvals = res[:,2]

    differences = np.abs(truevals - impvals)
    mae = np.mean(differences)
    print('average absolute error:', mae)

    res = ["${imputation}", mae]
    # Make pandas dataframe
    out_table = pd.DataFrame(res, index = ["imputation_strategy","MAE"])
    out_table["beta"] = "${beta}"

    # export table
    out_table.to_csv("${imputation}_beta${beta}_stats.csv", header = False,index=False)
    """
}
