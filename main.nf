nextflow.enable.dsl=2

/*
 * pipeline input parameters
 */
params.betaVAE_script = "$projectDir/betaVAEv2.py"
params.eval_sing_script = "$projectDir/nf_scripts/evaluate_single_imputation_eddie.py"
params.eval_mg_script = "$projectDir/nf_scripts/evaluate_metropolis_gibbs_eddie.py"
params.eval_pg_script = "$projectDir/nf_scripts/evaluate_pseudo_Gibbs_eddie.py"
params.eval_is_script = "$projectDir/nf_scripts/evaluate_importance_sampling_eddie.py"
params.configfile = "$projectDir/example_config_VAE.json"
params.lib_helper = "$projectDir/lib/helper_functions.py"

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
data_ch = channel.fromPath(params.data)
corrupt_data_ch = channel.fromPath(params.corrupt_data)
// scripts
betaVAE_ch = channel.fromPath(params.betaVAE_script, checkIfExists: true)
helper_ch = channel.fromPath(params.lib_helper, checkIfExists: true)
eval_sing_ch = channel.fromPath(params.eval_sing_script)
eval_mg_ch = channel.fromPath(params.eval_mg_script)
eval_pg_ch = channel.fromPath(params.eval_pg_script)
eval_is_ch = channel.fromPath(params.eval_is_script)
// config
config_ch = channel.fromPath(params.configfile)
// number of datasets
m_ch = Channel.of(1..40)

/*
 * train model
 * to do - read model settings of betaVAEv2 script from config that you pass in here
 */

process TRAIN_VAE {
    publishDir "${params.outdir}/model", mode: "copy"
    cpus 1
    memory '32 GB'

    input:
    path script
    path helper
    path config

    output:
    path('encoder.keras'), emit: encoder
    path('decoder.keras'), emit: decoder
    path('model_settings.json'), emit: model_settings
    path(script), emit: betaVAE

    script:
    """
    python $script --nextflow yes
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
    tuple val('single-imputatuion'), path('loglikelihood_across_iterations_single_imputed_dataset.csv'), emit: loglik

    script:
    """
    python $script --model $encoder
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
    tuple val('metropolis-within-gibbs'), path("loglikelihood_across_iterations_plaus_dataset_${dataset}.csv"), emit: loglik
    tuple val('metropolis-within-gibbs'), path("NA_imputed_values_plaus_dataset_${dataset}.csv"), emit: NAvals
    tuple val('metropolis-within-gibbs'), path("plaus_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --dataset $dataset
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
    tuple val('pseudo-gibbs'), path("loglikelihood_across_iterations_plaus_dataset_${dataset}.csv"), emit: loglik
    tuple val('pseudo-gibbs'), path("NA_imputed_values_plaus_dataset_${dataset}.csv"), emit: NAvals
    tuple val('pseudo-gibbs'), path("plaus_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --dataset $dataset
    """
}

process IMPUTE_MULTIPLE_iS {
    publishDir "${params.outdir}/multiple_imputation/importance-sampling", mode: "copy"
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
    path('importance_sampling_ESS.csv'), emit: ess
    tuple val('importance-sampling'), path('NA_imputed_values_plaus_dataset_*.csv'), emit: NAvals
    tuple val('importance-sampling'), path('plaus_dataset_*.csv'), emit: dataset

    script:
    """
    python $script --model $encoder --nDat $num_datasets
    """    
}


// main workflow
workflow {
    include { COMPILE_NA_INDICES; COMPUTE_CIs } from './modules/compile_stats.nf'
    include { LASSO } from './modules/downstream.nf'

    // number of datasets as single value for importance samping process
    m_dat=m_ch.count()

    // train VAE
    model=TRAIN_VAE(betaVAE_ch, helper_ch, config_ch)

    // run imputation strategies
    single_imp=SINGLE_IMPUTATION(model.betaVAE, eval_sing_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings)
    mult_mg=IMPUTE_MULTIPLE_MG(model.betaVAE, eval_mg_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_ch)
    mult_pg=IMPUTE_MULTIPLE_pG(model.betaVAE, eval_pg_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_ch)
    mult_is=IMPUTE_MULTIPLE_iS(model.betaVAE, eval_is_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_dat)
  
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

    // channel with all plausible datasets and imputation key
    all_dats=single_imp.dataset
                  .mix(mult_pg.dataset)
                  .mix(mult_pg.dataset)
                  .mix(mult_is.dataset)
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
                  .mix(mult_pg.dataset)
                  .mix(mult_pg.dataset)
                  .mix(mult_dat_flat)
                  .combine(corrupt_data_ch)
                  .combine(data_ch)
  
    LASSO(imp_dats)
}


