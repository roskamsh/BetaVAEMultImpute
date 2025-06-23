process TRAIN_VAE {
    publishDir "${params.outdir}/model/beta_${beta}", mode: "copy"
    cpus 1
    memory '10 GB'

    input:
    tuple val(beta), path(betaVAE), path(training_script), path(helper), path(config), path(data_complete), path(data_corrupt)

    output:
    tuple val(beta), path('encoder.keras'), path('decoder.keras'), path('model_settings.json')

    script:
    """
    python $training_script --config $config --nextflow True --beta $beta
    """
}

process SINGLE_IMPUTATION {
    publishDir "${params.outdir}/single-imputation/beta_${beta}", mode: "copy"
    cpus 1
    memory '10 GB'

    input:
    tuple val(beta), path(encoder), path(decoder), path(config), path(betaVAE), path(imputation_script), path(helper), path(data_complete), path(data_corrupt)

    output:
    tuple val(beta), val('single-imputation'), path('NA_imputed_values_single_imputed_dataset.csv'), emit: NAvals
    tuple val(beta), val('single-imputation'), path('single_imputed_dataset.csv'), emit: dataset
    tuple val(beta), val('single-imputation'), path('loglikelihood_across_iterations_single_imputed_dataset.csv'), emit: loglik

    script:
    """
    python $imputation_script --model $encoder --imputeBy si --outName single_imputed --config $config --nextflow True
    """
}

process IMPUTE_MULTIPLE_MG {
    publishDir "${params.outdir}/multiple-imputation/metropolis-within-gibbs/beta_${beta}", mode: "copy"
    cpus 1
    memory '10 GB'

    input:
    tuple val(beta), path(encoder), path(decoder), path(config), path(betaVAE), path(imputation_script), path(helper), path(data_complete), path(data_corrupt), val(dataset)

    output:
    tuple val(beta), val('metropolis-within-gibbs'), path("loglikelihood_across_iterations_mwg_dataset_${dataset}.csv"), emit: loglik
    tuple val(beta), val('metropolis-within-gibbs'), path("NA_imputed_values_mwg_dataset_${dataset}.csv"), emit: NAvals
    tuple val(beta), val('metropolis-within-gibbs'), path("mwg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $imputation_script --model $encoder --imputeBy mwg --dataset $dataset --outName mwg --config $config --nextflow True
    """
}

process IMPUTE_MULTIPLE_pG {
    publishDir "${params.outdir}/multiple-imputation/pseudo-gibbs/beta_${beta}", mode: "copy"
    cpus 1
    memory '10 GB'

    input:
    tuple val(beta), path(encoder), path(decoder), path(config), path(betaVAE), path(imputation_script), path(helper), path(data_complete), path(data_corrupt), val(dataset)

    output:
    tuple val(beta), val('pseudo-gibbs'), path("loglikelihood_across_iterations_pg_dataset_${dataset}.csv"), emit: loglik
    tuple val(beta), val('pseudo-gibbs'), path("NA_imputed_values_pg_dataset_${dataset}.csv"), emit: NAvals
    tuple val(beta), val('pseudo-gibbs'), path("pg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $imputation_script --model $encoder --imputeBy pg --dataset $dataset --outName pg --config $config --nextflow True
    """
}

process IMPUTE_MULTIPLE_iS {
    publishDir "${params.outdir}/multiple-imputation/sampling-importance-resampling/beta_${beta}", mode: "copy"
    cpus 1
    memory '10 GB'

    input:
    tuple val(beta), path(encoder), path(decoder), path(config), path(betaVAE), path(imputation_script), path(helper), path(data_complete), path(data_corrupt), val(num_datasets)

    output:
    path('sir_ESS.csv'), emit: ess
    tuple val(beta), val('sampling-importance-resampling'), path('NA_imputed_values_sir_dataset_*.csv'), emit: NAvals
    tuple val(beta), val('sampling-importance-resampling'), path('sir_dataset_*.csv'), emit: dataset
    tuple val(beta), val('sampling-importance-resampling'), path('MCMC_approximate_loglikelihood_SIR.csv'), emit: approx_loglik

    script:
    """
    python $imputation_script --model $encoder --imputeBy sir --nDat $num_datasets --outName sir --config $config \
        --nextflow True --sirProposal ${params.sir_proposal} --approx_loglik_numsamples_mcmc ${params.approx_loglik_numsamples_mcmc}
    """ 
}

process IMPUTE_MEAN {
    publishDir "${params.outdir}/mean-imputation", mode: "copy", pattern: "*mean*"
    cpus 1
    memory '10 GB'

    input:
    tuple path(helper), path(config), path(data_complete), path(data_corrupt)

    output:
    tuple val("none"), val("mean-imputation"), path("mean_imputed_dataset.csv"), path(data_corrupt), path(data_complete), emit: dataset 
    tuple val("none"), val("mean-imputation"), path("NA_imputed_values_mean-imputation.csv"), emit: NAvals

    script:
    """
    #!/usr/bin/env python
    from bin.helper_functions import get_scaled_data
    import json
    import numpy as np
    import pandas as pd

    configfile = "${config}"
    run_nextflow = True

    with open(configfile) as f:
        config = json.load(f)

    model_settings = \
        dict(
            latent_size=config["latent_size"],
            hidden_size_1=config["hidden_size_1"], 
            hidden_size_2=config["hidden_size_2"],  
            training_epochs = config["training_epochs"],
            batch_size = config["batch_size"],
            data_path = config["data_path"],
            corrupt_data_path = config["corrupt_data_path"],
            initial_imputation_strategy = config["initial_imputation_strategy"]
            )
    
    # Run initial imputation
    data, data_imputed = get_scaled_data(config["data_path"],config["corrupt_data_path"],
                                         initial_imputation_strategy="zero",
                                         nextflow=run_nextflow)
    # Get scaler & data_missing (with nans)
    data, data_missing, scaler = get_scaled_data(config["data_path"],config["corrupt_data_path"],
                                         initial_imputation_strategy="zero",
                                         nextflow=run_nextflow,
                                         return_scaler=True, put_nans_back=True)
    
    # Get missing sample rows
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    
    # Re-scale
    # mean-imputed data
    imputed_rescaled = scaler.inverse_transform(data_imputed.copy()) 
    imputed_missing_samples = imputed_rescaled[missing_rows].copy()
    # Original data
    data_rescaled = scaler.inverse_transform(data.copy())
    truevals_data_missing = data_rescaled[missing_rows].copy()

    # Export missing value indices
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], 'mean': imputed_missing_samples[na_ind]})
    na_indices.to_csv('NA_imputed_values_mean-imputation.csv')

    np.savetxt("mean_imputed_dataset.csv", imputed_missing_samples, delimiter=",")
    """
}