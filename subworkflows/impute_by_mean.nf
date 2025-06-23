include { IMPUTE_MEAN } from '../modules/train_and_impute.nf'
include { COMPUTE_MAE_SINGLE } from '../modules/compile_stats.nf'

workflow IMPUTE_BY_MEAN {
    main:
        // define input channels
        // data
        data_ch = channel.fromPath(params.data, checkIfExists: true)
        corrupt_data_ch = channel.fromPath(params.corrupt_data, checkIfExists: true)
        // scripts
        helper_ch = channel.fromPath(params.helper_bin, type: 'dir', checkIfExists: true)
        // config
        config_ch = channel.fromPath(params.configfile)

        input_ch = helper_ch
            .combine(config_ch)
            .combine(data_ch)
            .combine(corrupt_data_ch)

        if (params.run_mean_imputation) {
            IMPUTE_MEAN(input_ch) 
            mean_imputed_dataset = IMPUTE_MEAN.out.dataset
            mean_imputed_na_vals = IMPUTE_MEAN.out.NAvals
        } else {
            mean_imputed_dataset = Channel.empty()
            mean_imputed_na_vals = Channel.empty()
        }
        COMPUTE_MAE_SINGLE(mean_imputed_na_vals)
    emit:
        mean_imputed_dataset
}