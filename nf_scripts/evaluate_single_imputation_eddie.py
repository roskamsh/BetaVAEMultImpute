import os
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model
from lib.helper_functions import get_scaled_data, evaluate_coverage

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'output/model1/encoder.keras', type=str, help='directory which contains the trained VAE')

if __name__=="__main__":

    outname = 'single_imputed_dataset'
    print(outname)
    args = parser.parse_args()
    model_dir = '/'.join(args.model.split('/')[:-1]) + '/'
    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
   
    # impute by metropolis-within-Gibbs 
    missing_imputed, convergence_loglik = model.impute_single(data_corrupt=data_missing, data_complete = data, beta = 12, n_recycles=1000)

    # export output of m-th dataset
    data = scaler.inverse_transform(data)
    missing_imputed = scaler.inverse_transform(missing_imputed)

    truevals_data_missing = data[missing_rows]
    na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[na_ind]})

    na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
    np.savetxt(outname + ".csv", missing_imputed, delimiter=",")
    np.savetxt('loglikelihood_across_iterations_' + outname + '.csv', np.array(convergence_loglik), delimiter=',')

    print("Mean Absolute Error:", sum(((missing_imputed[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))
