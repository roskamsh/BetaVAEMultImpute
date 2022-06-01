import os
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
running_dir = os.getcwd()
# Add scripts in curent working directory to sys environment
sys.path.append(running_dir)

from betaVAEv2 import load_model
try:
    from lib.helper_functions import get_scaled_data, evaluate_coverage
except ModuleNotFoundError:
    from helper_functions import get_scaled_data, evaluate_coverage

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'encoder.keras', type=str, help='path to encoder.keras')
parser.add_argument('--nDat', type=int, default=1, help='number of datasets to are generating via MI for importance sampling')

if __name__=="__main__":
    args = parser.parse_args()

    # initialize input and set parameters for imputation
    model_dir = running_dir
    model = load_model(model_dir)
    max_iter = 3
    m_datasets = args.nDat

    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))

    # impute by importance sampling
    # output will be a list of all m datasets imputed by importance sampling (missing observations only)
    missing_imputed, ess = model.impute_multiple(data_corrupt=data_missing, max_iter=max_iter, 
                                                                m = m_datasets, 
                                                                method="importance sampling2")
    missing_imputed = np.array(missing_imputed)

    # export output of m-th dataset
    data = scaler.inverse_transform(data)
    truevals_data_missing = data[missing_rows]
    
    for i in range(m_datasets):
        outname = 'plaus_dataset_' + str(i+1)
        missing_imputed[i] = scaler.inverse_transform(missing_imputed[i])
        # export NA indices values
        na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[i][na_ind]})
        na_indices.to_csv('NA_imputed_values_' + outname + '.csv')
        # export each imputed dataset
        np.savetxt(outname + ".csv", missing_imputed[i], delimiter=",")
        print("Mean Absolute Error:", sum(((missing_imputed[i][na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0])) 

    np.savetxt('importance_sampling_ESS.csv', np.array(ess), delimiter=',')
