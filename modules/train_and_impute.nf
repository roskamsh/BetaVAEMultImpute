process TRAIN_VAE {
    publishDir "${params.outdir}/model/beta_${beta}", mode: "copy"
    cpus 1
    memory '32 GB'

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
    publishDir "${params.outdir}/single_imputation/beta_${beta}", mode: "copy"
    cpus 1
    memory '32 GB'

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
    publishDir "${params.outdir}/multiple_imputation/metropolis-within-gibbs/beta_${beta}", mode: "copy"
    cpus 1
    memory '32 GB'

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
    publishDir "${params.outdir}/multiple_imputation/pseudo-gibbs/beta_${beta}", mode: "copy"
    cpus 1
    memory '32 GB'

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
    publishDir "${params.outdir}/multiple_imputation/sampling-importance-resampling/beta_${beta}", mode: "copy"
    cpus 1
    memory '32 GB'

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