import os
import argparse
import pandas as pd
import numpy as np
import sys
# Add scripts in current working directory to sys environment
# This is necessary for now when running through nextflow and running code in job-specific hash directories
running_dir = os.getcwd()
sys.path.append(running_dir)

from betaVAE import load_model
from bin.helper_functions import get_scaled_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='encoder.keras', 
                    help='Path to trained VAE, in the same directory as decoder.keras')
parser.add_argument('--imputeBy', type=str, default='single_imputation',
                    help="""
                    Method to impute by.
                        Flags: 
                        (1) si - single imputation
                        (2) mwg - metropolis-within-gibbs 
                        (3) pg - pseudo-gibbs 
                        (4) sir - sampling importance resampling""")
parser.add_argument('--maxIter', type=int, default=1000,
                    help='Number of recycles for imputation')
parser.add_argument('--dataset', type=str, default='1', 
                    help='M-th dataset you are generating via multiple imputation. This should be specified if --nDat equals 1')
parser.add_argument('--nDat', type=int, default=1, 
                    help='Number of datasets to are generating via MI for importance sampling')
parser.add_argument('--outName', type=str, default='imputed',
                    help='Output name prefix for your imputed dataset')


if __name__=="__main__":

    args = parser.parse_args()
    outname = args.outName

    # Set model_dir
    if args.model.startswith('/'): # absolute path
        model_dir = os.path.split(args.model)[0]
    elif args.model.__contains__('/'): # relative path
        rel_path = os.path.split(args.model)[0]
        model_dir = os.path.join(running_dir,rel_path)
    else: # current working directory
        model_dir = running_dir

    # Load trained VAE
    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))

    # Only need to run impute_multiple() once if sampling importance resampling
    if args.imputeBy == 'sir':
        missing_imputed, ess = model.impute_multiple(data_corrupt=data_missing, max_iter=args.maxIter, m = args.nDat,
                                                                            method="sampling-importance-resampling")
        np.savetxt(outname + '_ESS.csv', np.array(ess), delimiter=',')
    # Re-scale data for comparing imputed values
    data_rescaled = scaler.inverse_transform(data.copy())
    truevals_data_missing = data_rescaled[missing_rows]

    # Impute M times
    for i in range(args.nDat):
        # Single imputation
        if args.imputeBy == 'si':
            if args.nDat > 1:
                sys.stderr.write('Single imputation specified, but nDat > 1. Please choose a multiple imputation method or specify nDat=1.\n')
                sys.exit(1)
            outname = args.outName
            missing_imputed, convergence_loglik = model.impute_single(data_corrupt=data_missing, data_complete = data, n_recycles=args.maxIter)            
            missing_imputed_rescaled = scaler.inverse_transform(missing_imputed.copy())
            na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed_rescaled[na_ind]})
            na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
            np.savetxt(outname + ".csv", missing_imputed_rescaled, delimiter=",")
            np.savetxt('loglikelihood_across_iterations_' + outname + '.csv', np.array(convergence_loglik), delimiter=',')
            print("Mean Absolute Error:", sum(((missing_imputed_rescaled[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))
        # Multiple imputation
        elif args.imputeBy in ['mwg','pg','sir']:
            # Only generating one dataset, use dataset argument to name the output file
            if args.nDat == 1:
                outname = args.outName + '_dataset_' + args.dataset
            # More than one dataset, use the nDat argument to name the output files
            else:
                outname = args.outName + '_dataset_' + str(i+1)
            if args.imputeBy == 'sir':
                missing_imputed[i] = scaler.inverse_transform(missing_imputed[i])
                na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[i][na_ind]})
                na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
                np.savetxt(outname + ".csv", missing_imputed[i], delimiter=",")
                print("Mean Absolute Error:", sum(((missing_imputed[i][na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))
            elif args.imputeBy in ['mwg','pg']:
                data_missing_copy = data_missing.copy()
                if args.imputeBy == 'mwg':
                    missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing_copy, max_iter=args.maxIter,
                                                                                method="Metropolis-within-Gibbs")
                elif args.imputeBy == 'pg':
                    missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing_copy, max_iter=args.maxIter,
                                                                                method="pseudo-Gibbs")
                missing_imputed_rescaled = scaler.inverse_transform(missing_imputed.copy())
                na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed_rescaled[na_ind]})
                na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
                np.savetxt(outname + ".csv", missing_imputed_rescaled, delimiter=",")
                np.savetxt('loglikelihood_across_iterations_' + outname + '.csv', np.array(convergence_loglik), delimiter=',')
                print("Mean Absolute Error:", sum(((missing_imputed_rescaled[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))                                          
            else:
                sys.stderr.write('No valid Multiple imputation procedure specified, but nDat > 1. Please refine nDat or specify --imputeBy to be mwg, pg or sir.\n') 
                sys.exit(1)
        else:
            sys.stderr.write('No valid imputation procedure specified. Please specify either si, mwg, pg or sir with the --imputeBy flag.\n')
            sys.exit(1)
