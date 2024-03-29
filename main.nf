nextflow.enable.dsl=2

/*
 * pipeline input parameters
 */
params.betaVAE = "$projectDir/betaVAE.py"
params.training_script = "$projectDir/train_VAE.py"
params.imputation_script = "$projectDir/impute_missing.py"
params.configfile = "$projectDir/VAE_config.json"
params.helper_bin = "$projectDir/bin"

println """\
         MULTIPLE IMPUTATION - NF PIPELINE
         ===================================
         complete data: ${params.data}
         corrupt data : ${params.corrupt_data}
         outdir       : ${params.outdir}
         """
         .stripIndent()


// define input channels
// data
data_ch = channel.fromPath(params.data, checkIfExists: true)
corrupt_data_ch = channel.fromPath(params.corrupt_data, checkIfExists: true)
// scripts
betaVAE_ch = channel.fromPath(params.betaVAE, checkIfExists: true)
helper_ch = channel.fromPath(params.helper_bin, type: 'dir', checkIfExists: true)
training_script_ch = channel.fromPath(params.training_script, checkIfExists: true)
imputation_script_ch = channel.fromPath(params.imputation_script, checkIfExists: true)
// config
config_ch = channel.fromPath(params.configfile)
// number of datasets
m_ch = Channel.of(1..100)

/*
 * train model
 * to do - read model settings of betaVAEv2 script from config that you pass in here
 */

process TRAIN_VAE {
    publishDir "${params.outdir}/model", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path betaVAE
    path script
    path helper
    path config

    output:
    path('encoder.keras'), emit: encoder
    path('decoder.keras'), emit: decoder
    path('model_settings.json'), emit: model_settings
    path(betaVAE), emit: betaVAE

    script:
    """
    python $script --config $config
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

    output:
    tuple val('single-imputation'), path('NA_imputed_values_single_imputed_dataset.csv'), emit: NAvals
    tuple val('single-imputation'), path('single_imputed_dataset.csv'), emit: dataset
    tuple val('single-imputation'), path('loglikelihood_across_iterations_single_imputed_dataset.csv'), emit: loglik

    script:
    """
    python $script --model $encoder --imputeBy si --outName single_imputed
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

    output:
    tuple val('metropolis-within-gibbs'), path("loglikelihood_across_iterations_mwg_dataset_${dataset}.csv"), emit: loglik
    tuple val('metropolis-within-gibbs'), path("NA_imputed_values_mwg_dataset_${dataset}.csv"), emit: NAvals
    tuple val('metropolis-within-gibbs'), path("mwg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy mwg --dataset $dataset --outName mwg
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

    output:
    tuple val('pseudo-gibbs'), path("loglikelihood_across_iterations_pg_dataset_${dataset}.csv"), emit: loglik
    tuple val('pseudo-gibbs'), path("NA_imputed_values_pg_dataset_${dataset}.csv"), emit: NAvals
    tuple val('pseudo-gibbs'), path("pg_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy pg --dataset $dataset --outName pg
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

    output:
    path('sir_ESS.csv'), emit: ess
    tuple val('sampling-importance-resampling'), path('NA_imputed_values_sir_dataset_*.csv'), emit: NAvals
    tuple val('sampling-importance-resampling'), path('sir_dataset_*.csv'), emit: dataset

    script:
    """
    python $script --model $encoder --imputeBy sir --nDat $num_datasets --outName sir
    """    
}


// main workflow
workflow {
    include { COMPILE_NA_INDICES; COMPUTE_CIs; COMPUTE_PERCENTILES; COMPUTE_MAE_SINGLE } from './modules/compile_stats.nf'
    include { LASSO; LASSO_TRUE } from './modules/downstream.nf'

    // number of datasets as single value for importance samping process
    m_dat=m_ch.count()

    // train VAE
    model=TRAIN_VAE(betaVAE_ch, training_script_ch, helper_ch, config_ch)

    // run imputation strategies
    single_imp=SINGLE_IMPUTATION(model.betaVAE, imputation_script_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings)
    mult_mg=IMPUTE_MULTIPLE_MG(model.betaVAE, imputation_script_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_ch)
    mult_pg=IMPUTE_MULTIPLE_pG(model.betaVAE, imputation_script_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_ch)
    mult_is=IMPUTE_MULTIPLE_iS(model.betaVAE, imputation_script_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_dat)
  
    // channel with all NAvals files for each imputation strategy, 1 emission per strategy
    NAvals_ch = mult_mg.NAvals
                   .mix(mult_pg.NAvals)
                   .mix(mult_is.NAvals)
                   .groupTuple()
                   .map {
                       group, files ->
                       [group, files.flatten()]
                   }

    comp_na=COMPILE_NA_INDICES(NAvals_ch)

    COMPUTE_CIs(comp_na)
    COMPUTE_MAE_SINGLE(single_imp.NAvals)

    // mix together all imputation NA index results for computing percentiles
    COMPUTE_PERCENTILES(comp_na) 

    // configure channel with all plausible datasets and imputation key
    // importance sampling output looks different than the other two so need to reformat
    mult_is.dataset
             .multiMap {
                group, files -> 
                  key: [group]
                  files: files.flatten()
             }
             .set {split_is_out}
    // re-combine the split up channels so you have key, file as a unique emission per dataset
    mult_dat_flat = split_is_out.key.combine(split_is_out.files.flatten())

    // channel with all plausible datasets and imputation key
    imp_dats=single_imp.dataset
                  .mix(mult_mg.dataset)
                  .mix(mult_pg.dataset)
                  .mix(mult_dat_flat)
                  .combine(corrupt_data_ch)
                  .combine(data_ch)

    LASSO(imp_dats)

    LASSO_TRUE(data_ch)
}


