include { LASSO; LASSO_TRUE } from '../modules/downstream.nf'

workflow EVALUATE_LASSO {
    take:
        imp_dats 
        data_ch
    main:
        LASSO(imp_dats)

        LASSO_TRUE(data_ch)
}