process TRAIN_VAE {
    publishDir "${params.outdir}/model", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config
    path data_complete
    path data_corrupt

    output:
    path('encoder.keras'), emit: encoder
    path('decoder.keras'), emit: decoder
    path('model_settings.json'), emit: model_settings
    path(betaVAE), emit: betaVAE

    script:
    """
    python $script --config $config --nextflow True
    """
}

process SINGLE_IMPUTATION {
    publishDir "${params.outdir}/single_imputation", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config
    path encoder
    path decoder
    path model_settings
    path data_complete
    path data_corrupt

    output:
    tuple val('single-imputation'), path('NA_imputed_values_single_imputed_dataset.csv'), emit: NAvals
    tuple val('single-imputation'), path('single_imputed_dataset.csv'), emit: dataset
    tuple val('single-imputation'), path('loglikelihood_across_iterations_single_imputed_dataset.csv'), emit: loglik

    script:
    """
    python $script --model $encoder --imputeBy si --outName single_imputed --config $config
    """
}

process IMPUTE_MULTIPLE_MG {
    publishDir "${params.outdir}/multiple_imputation/metropolis-within-gibbs", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config
    path encoder
    path decoder
    path model_settings
    each dataset
    path data_complete
    path data_corrupt

    output:
    tuple val('metropolis-within-gibbs'), path("loglikelihood_across_iterations_mwg_dataset_${dataset}.csv"), emit: loglik
    tuple val('metropolis-within-gibbs'), path("NA_imputed_values_mwg_dataset_${dataset}.csv"), emit: NAvals
    tuple val('metropolis-within-gibbs'), path("mwg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy mwg --dataset $dataset --outName mwg --config $config
    """
}

process IMPUTE_MULTIPLE_pG {
    publishDir "${params.outdir}/multiple_imputation/pseudo-gibbs", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config
    path encoder
    path decoder
    path model_settings
    each dataset
    path data_complete
    path data_corrupt

    output:
    tuple val('pseudo-gibbs'), path("loglikelihood_across_iterations_pg_dataset_${dataset}.csv"), emit: loglik
    tuple val('pseudo-gibbs'), path("NA_imputed_values_pg_dataset_${dataset}.csv"), emit: NAvals
    tuple val('pseudo-gibbs'), path("pg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy pg --dataset $dataset --outName pg --config $config
    """
}

process IMPUTE_MULTIPLE_iS {
    publishDir "${params.outdir}/multiple_imputation/sampling-importance-resampling", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config
    path encoder
    path decoder
    path model_settings
    val num_datasets
    path data_complete
    path data_corrupt

    output:
    path('sir_ESS.csv'), emit: ess
    tuple val('sampling-importance-resampling'), path('NA_imputed_values_sir_dataset_*.csv'), emit: NAvals
    tuple val('sampling-importance-resampling'), path('sir_dataset_*.csv'), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy sir --nDat $num_datasets --outName sir --config $config
    """    
}