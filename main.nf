nextflow.enable.dsl=2

// Default parameters
params.outdir = "${launchDir}/output"
params.pipeline_report_dir = "${launchDir}/logs"
params.m = 100 // Number of datasets to impute with multiple imputation
params.sir_proposal = "t" // can be "t" or "normal"
params.approx_loglik_numsamples_mcmc = 1000
params.coverage_levels_to_plot = [90,95]

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
include { TRAIN_AND_IMPUTE } from './subworkflows/train_and_impute.nf'
include { EVALUATE_LASSO } from './subworkflows/downstream.nf'
include { PLOT_ACROSS_BETAS } from './modules/plot.nf'

// workflow to tune beta
workflow TUNE_BETA {
    run_single_imputation = false
    beta_ch = Channel.fromList(params.betas_to_check)
    coverage_levels_to_plot = Channel.fromList(params.coverage_levels_to_plot)
        .collect()
        .map{ list -> [list]}
    
    TRAIN_AND_IMPUTE(beta_ch, run_single_imputation)

    TRAIN_AND_IMPUTE.out.mae_cis
        .join(TRAIN_AND_IMPUTE.out.approx_loglik)
        .map { beta_value, stats, loglik -> [[beta_value, stats, loglik]]}
        .collect()
        .map { all_tuples ->
            return all_tuples.transpose()
        }
        .combine(coverage_levels_to_plot)
        .set { plot_input_ch }

    PLOT_ACROSS_BETAS(plot_input_ch)
}

// main workflow
workflow {
    println """\
         MULTIPLE IMPUTATION - NF PIPELINE
         ===================================
         complete data: ${params.data}
         corrupt data : ${params.corrupt_data}
         outdir       : ${params.outdir}
         """.stripIndent()
    
    beta_ch = Channel.of(params.beta)
    run_single_imputation = true
    TRAIN_AND_IMPUTE(beta_ch, run_single_imputation)
    
    EVALUATE_LASSO(TRAIN_AND_IMPUTE.out.imputed_datasets, TRAIN_AND_IMPUTE.out.complete_data)
}
