import os
import argparse
import pandas as pd
import numpy as np
import sys
running_dir = os.getcwd()
# Add scripts in current working directory to sys environment
# This is necessary for now when running through nextflow and running code in job-specific hash directories
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
                    help='M-th dataset you are generating via multiple imputation')
parser.add_argument('--nDat', type=int, default=1, 
                    help='Number of datasets to are generating via MI for importance sampling')
parser.add_argument('--outName', type=str, default='imputed',
                    help='Output name prefix for your imputed dataset')


if __name__=="__main__":

    args = parser.parse_args()
    outname = args.outName

    # Set model_dir
    if args.model.startswith('/'):
        model_dir = os.path.split(args.model)[0]
    elif args.model.__contains__('/'):
        rel_path = os.path.split(args.model)[0]
        model_dir = os.path.join(running_dir,rel_path)
    else:
        model_dir = running_dir

    # Load trained VAE
    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
   
    # Impute
    if args.imputeBy =='si':
        missing_imputed, convergence_loglik = model.impute_single(data_corrupt=data_missing, data_complete = data, n_recycles=args.maxIter)
    elif args.imputeBy == 'mwg':
        missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing, max_iter=args.maxIter,
                                                                                method="Metropolis-within-Gibbs")
    elif args.imputeBy == 'pg':
        missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing, max_iter=args.maxIter,
                                                                                method="pseudo-Gibbs")
    elif args.imputeBy == 'sir':
        missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing, max_iter=args.maxIter,
                                                                                method="sampling-importance-resampling")
    else:
        sys.err('No valid imputation method provided. Please specify either si, mwg, pg or sir.')
    
    # Re-scale data
    data = scaler.inverse_transform(data)
    missing_imputed = scaler.inverse_transform(missing_imputed)

    # Export output
    truevals_data_missing = data[missing_rows]
    na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[na_ind]})

    na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
    np.savetxt(outname + ".csv", missing_imputed, delimiter=",")
    np.savetxt('loglikelihood_across_iterations_' + outname + '.csv', np.array(convergence_loglik), delimiter=',')

    print("Mean Absolute Error:", sum(((missing_imputed[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))
