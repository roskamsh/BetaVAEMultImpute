// Modules to import
include { TRAIN_VAE; SINGLE_IMPUTATION; IMPUTE_MULTIPLE_MG; IMPUTE_MULTIPLE_pG; IMPUTE_MULTIPLE_iS } from '../modules/train_and_impute.nf'
include { COMPILE_NA_INDICES; COMPUTE_CIs; COMPUTE_PERCENTILES; COMPUTE_MAE_SINGLE; SUMMARISE_APPROX_LOGLIK } from '../modules/compile_stats.nf'

// main workflow
workflow TRAIN_AND_IMPUTE {
    take:
        beta
        run_single_imputation
    main:
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
        m_ch=Channel.of(1..params.m) // for testing purposes
        // number of datasets as single value for importance samping process
        m_count=m_ch.count()

        // train VAE
        input_ch = beta
            .combine(betaVAE_ch)
            .combine(training_script_ch)
            .combine(helper_ch)
            .combine(config_ch)
            .combine(data_ch)
            .combine(corrupt_data_ch)
        model=TRAIN_VAE(input_ch)

        imputation_input_ch = model
            .combine(betaVAE_ch) 
            .combine(imputation_script_ch)
            .combine(helper_ch)
            .combine(data_ch)
            .combine(corrupt_data_ch)
        
        // run imputation strategies
        if (run_single_imputation) {
            single_imp=SINGLE_IMPUTATION(imputation_input_ch)
            single_imp_dataset = single_imp.dataset
            COMPUTE_MAE_SINGLE(single_imp.NAvals)
        } else {
            single_imp_dataset = Channel.empty()
        }

        mwg_pg_input_ch = imputation_input_ch 
            .combine(m_ch)
        sir_input_ch = imputation_input_ch
            .combine(m_count)

        // Run Metropolis-within-Gibbs if specified
        if (params.run_mwg == true) {
            mult_mg=IMPUTE_MULTIPLE_MG(mwg_pg_input_ch)
            mult_mg_NA_vals = mult_mg.NAvals
            mult_mg_dataset = mult_mg.dataset
        } else {
            mult_mg_NA_vals = Channel.empty()
            mult_mg_dataset = Channel.empty()
        }
        
        // Run Pseudo-Gibbs if specified
        if (params.run_pg == true) { 
            mult_pg=IMPUTE_MULTIPLE_pG(mwg_pg_input_ch)
            mult_pg_NA_vals = mult_pg.NAvals
            mult_pg_dataset = mult_pg.dataset
        } else {
            mult_pg_NA_vals = Channel.empty()
            mult_pg_dataset = Channel.empty()
        }

        // Always run sampling importance resampling
        mult_is=IMPUTE_MULTIPLE_iS(sir_input_ch)

        // channel with all NAvals files for each imputation strategy, 1 emission per strategy
        NAvals_ch = mult_mg_NA_vals
                    .mix(mult_pg_NA_vals)
                    .mix(mult_is.NAvals)
                    .groupTuple(by: [0,1])
                    .map {
                        beta_value, imputation, files ->
                        [beta_value, imputation, files.flatten()]
                    }

        comp_na=COMPILE_NA_INDICES(NAvals_ch)

        COMPUTE_CIs(comp_na)

        // mix together all imputation NA index results for computing percentiles
        COMPUTE_PERCENTILES(comp_na) 

        // Summarise approximate loglikelihood
        SUMMARISE_APPROX_LOGLIK(mult_is.approx_loglik)

        // configure channel with all plausible datasets and imputation key
        // importance sampling output looks different than the other two so need to reformat
        mult_is.dataset.flatMap { beta_value, imputation, files ->
                files.collect { file ->
                    [beta_value, imputation, file]
                }
            }
            .set{ mult_dat_flat }

        // channel with all plausible datasets and imputation key
        imp_dats=single_imp_dataset
                    .mix(mult_mg_dataset)
                    .mix(mult_pg_dataset)
                    .mix(mult_dat_flat)
                    .combine(corrupt_data_ch)
                    .combine(data_ch)

    emit:
        imputed_datasets = imp_dats
        complete_data = data_ch
        mae_cis = COMPUTE_CIs.out
        approx_loglik = SUMMARISE_APPROX_LOGLIK.out
}