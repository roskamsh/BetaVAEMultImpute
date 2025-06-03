nextflow.enable.dsl=2

// Default parameters
params.outdir = "${launchDir}/output"
params.pipeline_report_dir = "${launchDir}/logs"

// Pipeline input parameters
params.betaVAE = "${projectDir}/betaVAE.py"
params.training_script = "${projectDir}/train_VAE.py"
params.imputation_script = "${projectDir}/impute_missing.py"
params.configfile = "You must provide this file" // Example provided at VAE_config.json
params.helper_bin = "${projectDir}/bin"

// Input data
params.data = "You must provide the complete dataset"
params.corrupt_data = "You must provide the corrupt dataset"

// Optional: do not run certain imputation approaches
params.run_mwg = true
params.run_pg = true
params.run_sir = true

// Modules to import
include { TRAIN_VAE; SINGLE_IMPUTATION; IMPUTE_MULTIPLE_MG; IMPUTE_MULTIPLE_pG; IMPUTE_MULTIPLE_iS } from './modules/train_and_impute.nf'
include { COMPILE_NA_INDICES; COMPUTE_CIs; COMPUTE_PERCENTILES; COMPUTE_MAE_SINGLE } from './modules/compile_stats.nf'
include { LASSO; LASSO_TRUE } from './modules/downstream.nf'

// main workflow
workflow {
    println """\
         MULTIPLE IMPUTATION - NF PIPELINE
         ===================================
         complete data: ${params.data}
         corrupt data : ${params.corrupt_data}
         outdir       : ${params.outdir}
         """.stripIndent()
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
    //m_ch = Channel.of(1..100)
    m_ch=Channel.of(1..3) // for testing purposes
    // number of datasets as single value for importance samping process
    m_dat=m_ch.count()

    // train VAE
    model=TRAIN_VAE(betaVAE_ch, training_script_ch, helper_ch, config_ch, data_ch, corrupt_data_ch)

    // run imputation strategies
    single_imp=SINGLE_IMPUTATION(
        model.betaVAE, imputation_script_ch, helper_ch, 
        config_ch, model.encoder, model.decoder, 
        model.model_settings, data_ch, corrupt_data_ch
    )

    // Run Metropolis-within-Gibbs if specified
    if (params.run_mwg == true) {
        mult_mg=IMPUTE_MULTIPLE_MG(
            model.betaVAE, imputation_script_ch, helper_ch, 
            config_ch, model.encoder, model.decoder, 
            model.model_settings, m_ch, data_ch, corrupt_data_ch
        )
        mult_mg_NA_vals = mult_mg.NAvals
        mult_mg_dataset = mult_mg.dataset
    } else {
        mult_mg_NA_vals = Channel.empty()
        mult_mg_dataset = Channel.empty()
    }
    
    // Run Pseudo-Gibbs if specified
    if (params.run_pg == true) { 
        mult_pg=IMPUTE_MULTIPLE_pG(
            model.betaVAE, imputation_script_ch, helper_ch, 
            config_ch, model.encoder, model.decoder, 
            model.model_settings, m_ch,  data_ch, corrupt_data_ch
        )
        mult_pg_NA_vals = mult_pg.NAvals
        mult_pg_dataset = mult_pg.dataset
    } else {
        mult_pg_NA_vals = Channel.empty()
        mult_pg_dataset = Channel.empty()
    }

    // Always run sampling importance resampling
    mult_is=IMPUTE_MULTIPLE_iS(
        model.betaVAE, imputation_script_ch, helper_ch, 
        config_ch, model.encoder, model.decoder, 
        model.model_settings, m_dat,  data_ch, corrupt_data_ch
    )
  
    // channel with all NAvals files for each imputation strategy, 1 emission per strategy
    NAvals_ch = mult_mg_NA_vals
                   .mix(mult_pg_NA_vals)
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
                  .mix(mult_mg_dataset)
                  .mix(mult_pg_dataset)
                  .mix(mult_dat_flat)
                  .combine(corrupt_data_ch)
                  .combine(data_ch)

    LASSO(imp_dats)

    LASSO_TRUE(data_ch)
    
}
