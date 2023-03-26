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
    ci_70 = 1.036
    ci_75 = 1.150
    ci_80 = 1.282
    ci_85 = 1.440
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_70 = sum(n_deviations < ci_70) / len(n_deviations)
    prop_75 = sum(n_deviations < ci_75) / len(n_deviations)
    prop_80 = sum(n_deviations < ci_80) / len(n_deviations)
    prop_85 = sum(n_deviations < ci_85) / len(n_deviations)
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    print('prop 70:', prop_70)
    print('prop 75:', prop_75)
    print('prop 80:', prop_80)
    print('prop 85:', prop_85)
    print('prop 90:', prop_90)
    print('prop 95:', prop_95)
    print('prop 99:', prop_99)

    differences = np.abs(truevals - means)
    mae = np.mean(differences)
    print('average absolute error:', mae)

    res = ["${imputation}", mae, prop_70, prop_75, prop_80, prop_85, prop_90, prop_95, prop_99]

    # Make pandas dataframe
    out_table = pd.DataFrame(res, index = ["imputation_strategy","MAE","ci_70","ci_75","ci_80","ci_85","ci_90","ci_95","ci_99"])

    # export table
    out_table.to_csv("${imputation}_stats.csv", header = False)
    """
}

process COMPUTE_PERCENTILES {
    publishDir "${params.outdir}/multiple_imputation", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    tuple val(imputation), path(na_indices)

    output:
    path("${imputation}_imputation_percentiles.csv")

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

    # export table
    out_table.to_csv("${imputation}_imputation_percentiles.csv", header = True) 
    """
}

process COMPUTE_MAE_SINGLE {
    publishDir "${params.outdir}/single_imputation", mode: "copy"
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
    truevals = res[:,1]
    impvals = res[:,2]

    differences = np.abs(truevals - impvals)
    mae = np.mean(differences)
    print('average absolute error:', mae)

    res = ["${imputation}", mae]
    # Make pandas dataframe
    out_table = pd.DataFrame(res, index = ["imputation_strategy","MAE"])

    # export table
    out_table.to_csv("${imputation}_stats.csv", header = False)
    """
}
