nextflow.enable.dsl=2

/*
 * pipeline input parameters
 */
params.betaVAE_script = "$projectDir/betaVAEv2.py"
params.eval_sing_script = "$projectDir/nf_scripts/evaluate_single_imputation_eddie.py"
params.eval_mg_script = "$projectDir/nf_scripts/evaluate_metropolis_gibbs_eddie.py"
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
    file betaVAE
    file script
    file helper
    file config
    file encoder
    file decoder
    file model_settings

    output:
    file 'NA_imputed_values_single_imputed_dataset.csv'
    file 'single_imputed_dataset.csv'
    file 'loglikelihood_across_iterations_single_imputed_dataset.csv'

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
    file betaVAE
    file script
    file helper
    file config
    file encoder
    file decoder
    file model_settings
    each dataset

    output:
    path("loglikelihood_across_iterations_plaus_dataset_${dataset}.csv"), emit: loglik
    path("NA_imputed_values_plaus_dataset_${dataset}.csv"), emit: NAvals
    path("plaus_dataset_${dataset}.csv"), emit: dataset

    script:
    """
    python $script --model $encoder --dataset $dataset
    """
}

workflow {
  model=TRAIN_VAE(betaVAE_ch, helper_ch, config_ch)
  single_imp=SINGLE_IMPUTATION(model.betaVAE, eval_sing_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings)
  mult_mg=IMPUTE_MULTIPLE_MG(model.betaVAE, eval_mg_ch, helper_ch, config_ch, model.encoder, model.decoder, model.model_settings, m_ch)

}


