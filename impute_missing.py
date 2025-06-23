import os
import argparse
import pandas as pd
import numpy as np
import sys
import json
# Add scripts in current working directory to sys environment
# This is necessary for now when running through nextflow and running code in job-specific hash directories
running_dir = os.getcwd()
sys.path.append(running_dir)

from betaVAE import load_model
from bin.helper_functions import get_scaled_data, log_lik_ymis_given_obs_mcmc_q, log_lik_ymis_given_obs_mcmc_p

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='path to configuration json file')
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
parser.add_argument('--sirProposal', type=str, default='t', help = "Proposal distribution to use for sampling-importance-resampling")
parser.add_argument('--approx_loglik_numsamples_mcmc', type=int, default=1000, help = "Number of samples for MCMC approximation while computing p(ymis|yobs)")
parser.add_argument('--outName', type=str, default='imputed',
                    help='Output name prefix for your imputed dataset')
parser.add_argument('--nextflow', type=bool, default=False, help = 'Whether you are running from a nextflow pipeline or not.')


if __name__=="__main__":

    args = parser.parse_args()
    outname = args.outName
    configfile = args.config
    model_file = args.model 
    imputeby = args.imputeBy
    max_iter = args.maxIter 
    dataset = args.dataset 
    proposal = args.sirProposal
    n_dat = args.nDat 
    outname = args.outName
    run_nextflow = args.nextflow
    S = args.approx_loglik_numsamples_mcmc

    with open(configfile) as f:
        config = json.load(f)

    # Set model_dir
    if model_file.startswith('/'): # absolute path
        model_dir = os.path.split(model_file)[0]
    elif model_file.__contains__('/'): # relative path
        rel_path = os.path.split(model_file)[0]
        model_dir = os.path.join(running_dir,rel_path)
    else: # current working directory
        model_dir = running_dir

    # Load trained VAE
    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(config["data_path"],config["corrupt_data_path"],
                                                 initial_imputation_strategy=config["initial_imputation_strategy"],
                                                 put_nans_back=True, return_scaler=True, nextflow=run_nextflow)
    
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))

    # Only need to run impute_multiple() once if sampling importance resampling
    if imputeby == 'sir':
        # max_iter here is the number of samples, S, that we take
        missing_imputed, ess = model.impute_multiple(
            data_corrupt=data_missing, max_iter=max_iter, m = n_dat,
            method="sampling-importance-resampling", proposal=proposal
        )
        np.savetxt(outname + '_ESS.csv', np.array(ess), delimiter=',')
        mcmc_p = log_lik_ymis_given_obs_mcmc_p(data, data_missing, model, num_samples_mc=S)
        mcmc_q = log_lik_ymis_given_obs_mcmc_q(data, data_missing, model, num_samples_mc=S, proposal = proposal, df = 3)
        approx_loglik = pd.DataFrame({'mcmc_p': mcmc_p, 'mcmc_q': mcmc_q})
        approx_loglik.to_csv(f"MCMC_approximate_loglikelihood_SIR.csv",index=False)
    
    # Re-scale data for comparing imputed values
    data_rescaled = scaler.inverse_transform(data.copy())
    truevals_data_missing = data_rescaled[missing_rows]

    # Impute M times
    for i in range(n_dat):
        # Single imputation
        if imputeby == 'si':
            if n_dat > 1:
                sys.stderr.write('Single imputation specified, but nDat > 1. Please choose a multiple imputation method or specify nDat=1.\n')
                sys.exit(1)
            outname_i = outname + '_dataset'
            missing_imputed, convergence_loglik = model.impute_single(data_corrupt=data_missing, data_complete = data, n_recycles=max_iter)            
            missing_imputed_rescaled = scaler.inverse_transform(missing_imputed.copy())
            na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed_rescaled[na_ind]})
            na_indices.to_csv('NA_imputed_values_' + outname_i + '.csv')
            mae = sum(((missing_imputed_rescaled[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]) 
            mae_df = pd.DataFrame({'dataset': [outname_i], 'MAE': [mae]})
            mae_df.to_csv(f"MAE_{outname_i}.csv", index=False)
            np.savetxt(outname_i + ".csv", missing_imputed_rescaled, delimiter=",")
            np.savetxt('loglikelihood_across_iterations_' + outname_i + '.csv', np.array(convergence_loglik), delimiter=',')
            print(f"Mean Absolute Error: {mae}")
        # Multiple imputation
        elif imputeby in ['mwg','pg','sir']:
            # Only generating one dataset, use dataset argument to name the output file
            if n_dat == 1:
                outname_i = outname + '_dataset_' + dataset
            # More than one dataset, use the nDat argument to name the output files
            else:
                outname_i = outname + '_dataset_' + str(i+1)
            if imputeby == 'sir':
                missing_imputed[i] = scaler.inverse_transform(missing_imputed[i])
                na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[i][na_ind]})
                na_indices.to_csv('NA_imputed_values_' + outname_i + '.csv')
                np.savetxt(outname_i + ".csv", missing_imputed[i], delimiter=",")
                mae = sum(((missing_imputed[i][na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0])
                mae_df = pd.DataFrame({'dataset': [outname_i], 'MAE': [mae]})
                mae_df.to_csv(f"MAE_{outname_i}.csv", index=False)
                print(f"Mean Absolute Error: {mae}")
            elif imputeby in ['mwg','pg']:
                data_missing_copy = data_missing.copy()
                if imputeby == 'mwg':
                    missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing_copy, max_iter=max_iter,
                                                                                method="Metropolis-within-Gibbs")
                elif imputeby == 'pg':
                    missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing_copy, max_iter=max_iter,
                                                                                method="pseudo-Gibbs")
                missing_imputed_rescaled = scaler.inverse_transform(missing_imputed.copy())
                na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed_rescaled[na_ind]})
                na_indices.to_csv('NA_imputed_values_' + outname_i + '.csv')
                np.savetxt(outname_i + ".csv", missing_imputed_rescaled, delimiter=",")
                np.savetxt('loglikelihood_across_iterations_' + outname_i + '.csv', np.array(convergence_loglik), delimiter=',')
                mae = sum(((missing_imputed_rescaled[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0])
                mae_df = pd.DataFrame({'dataset': [outname_i], 'MAE': [mae]})
                mae_df.to_csv(f"MAE_{outname_i}.csv", index=False)
                print("Mean Absolute Error:", mae)                                          
            else:
                raise ValueError(f"No valid Multiple imputation procedure specified, but nDat > 1. Please refine nDat or specify --imputeBy to be mwg, pg or sir.")

        else:
            raise ValueError(f"No valid imputation procedure specified. Please specify either si, mwg, pg or sir with the --imputeBy flag.")
