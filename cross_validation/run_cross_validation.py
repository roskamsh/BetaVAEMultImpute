import sys
sys.path.append('/SAN/orengolab/nsp13/BetaVAEMImputation')
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from bin.helper_functions import get_scaled_data, DataMissingMaker, evaluate_coverage, evaluate_coverage_quantile
from betaVAE import VariationalAutoencoder
import tensorflow as tf


"""
This script runs cross-validation in order to tune the value of beta and the number of epochs.
Additional missingness is added to the original data, and the accuracy and uncertainty
are measured based on how well they reconstruct the missing data that was deliberately introduced.
The dataset it copied 'k_fold' times so that each of the copies does not have too much missingness
introduced.

This script is designed for parallelization where each parallel run tests a different value of beta
the test value of beta is indexed by d_index. 
"""

class CrossValidation:
    """Creates an object to run cross-validation over values for epochs and beta.
    """
    def __init__(self, model, config):
        pass


def evaluate_variance(model, missing_w_nans, na_ind):
    missing_w_zeros = np.nan_to_num(missing_w_nans)
    x_hat_mean, x_hat_log_sigma_sq = model.predict(missing_w_zeros)
    return np.mean(x_hat_log_sigma_sq.numpy()[na_ind])

def generate_multiple_and_evaluate_coverage(model, missing_w_nans, missing_complete, na_ind, scaler, recycles, m):
    multi_imputes_missing =[]
    m_datasets = m
    missing_row_ind = np.where(np.isnan(missing_w_nans).any(axis=1))
    subset_na = np.where(np.isnan(missing_w_nans[missing_row_ind]))
    for i in range(m_datasets):
        missing_imputed, convergence_loglik = model.impute_multiple(missing_w_nans, max_iter=recycles, method = "Metropolis-within-Gibbs")
        multi_imputes_missing.append(missing_imputed[subset_na])
    results_quantile = evaluate_coverage_quantile(multi_imputes_missing, missing_complete, missing_w_nans, scaler)
    results  = evaluate_coverage(multi_imputes_missing, missing_complete, missing_w_nans, scaler)
    results.update(results_quantile)
    return results

def evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler, recycles, m):
    coverage_results = generate_multiple_and_evaluate_coverage(model, np.copy(missing_w_nans), missing_complete, na_ind, scaler, recycles, m)
    _, _, all_mae = model.impute_single(np.copy(missing_w_nans), missing_complete, n_recycles=6, loss='MAE', scaler=scaler, return_losses=True)
    results = dict(
    mae = all_mae[-1],
    average_variance = evaluate_variance(model, missing_w_nans, na_ind)
    )
    for k,v in coverage_results.items():
        results[k] = v
    return results

def get_additional_masked_data(complete_w_nan, prop_miss_rows=1, prop_miss_col=0.1):
    complete_row_index = np.where(np.isfinite(complete_w_nan).all(axis=1))[0]
    complete_only = complete_w_nan[complete_row_index]
    miss_maker = DataMissingMaker(complete_only, prop_miss_rows=prop_miss_rows, prop_miss_col=prop_miss_col)
    extra_missing_validation =  miss_maker.generate_missing_data()
    # assert np.isnan(extra_missing_validation, axis=0)
    val_na_ind = np.where(np.isnan(extra_missing_validation))
    return extra_missing_validation, complete_only, val_na_ind

def create_lock(path='lock.txt'):
    with open(path, 'w') as filehandle:
        filehandle.write('temp lock')

def remove_lock(path='lock.txt'):
    os.remove(path)

def save_results(results, epoch, beta, results_path='beta_analysis.csv', lock_path='lock.txt'):
    if not os.path.exists(results_path):
        with open(results_path, 'w') as filehandle:
            filehandle.write('beta,epoch,mae,multi_mae,average_variance,prop_90,prop_95,prop_99\n')
    while os.path.exists(lock_path):
        print('sleeping due to file lock') # prevent paralel runs from writing to the file at the same time
        time.sleep(2)
    create_lock()
    df = pd.read_csv(results_path)
    results['epoch'] = epoch
    results['beta'] = beta
    df = df.append(results, ignore_index=True)
    df.to_csv(results_path, index=False)

def split_training_and_validation(config, k, data_missing_nan):
    n_per_fold = len(data) // config['k_folds']
    start_index = k * n_per_fold
    if k == config['k_folds'] - 1:
        # if on the last fold then go to the end of the data
        end_index = len(data)
    else:
        end_index = start_index + n_per_fold
    current_fold = data_missing_nan[start_index:end_index]
    test_missing_row_ind = np.where(np.isnan(data_missing_nan).any(axis=1))[0]
    val_missing_row_ind = list(set(range(start_index, end_index)) - set(test_missing_row_ind))

    validation_w_nan, validation_complete, val_na_ind = get_additional_masked_data(current_fold, prop_miss_rows=1,
                                                                                   prop_miss_col=0.1)
    training_input = np.copy(data_missing_nan)
    training_input[val_missing_row_ind] = validation_w_nan
    training_input = np.nan_to_num(training_input)
    return training_input, validation_w_nan, validation_complete, val_na_ind

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('d_index', type=int, help='index of the run')
    parser.add_argument('--config', type=str, default='cross_validation/cv_config.json', help='path to configuration json file')
    args = parser.parse_args()
    d_index = args.d_index -1
    with open(args.config) as f:
        config = json.load(f)

    model_settings = \
        dict(
            latent_size=config["latent_size"],
            hidden_size_1=config["hidden_size_1"],
            hidden_size_2=config["hidden_size_2"],
            training_epochs = config["training_epochs"],
            batch_size = config["batch_size"],
            beta = config["beta"],
            data_path = config["data_path"],
            corrupt_data_path = config["corrupt_data_path"]
            )
    data, data_missing_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    n_row = data.shape[1]
    model_settings['input_size']=n_row
    epoch_granularity = {0.1: 15, 0.5: 20, 1: 20, 1.25: 20, 1.5: 25, 1.75: 25, 2: 25, 2.5: 30, 3: 30, 4: 30, 5: 30,
                         6: 30, 8: 30, 12: 30, 16: 30, 24: 30, 32: 35, 50: 40, 64: 50, 100: 100, 150: 100}

    beta_rates = [0.1, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 12, 16, 24, 32, 50, 64]
    n_epochs_dict = {0.1: 300, 0.5: 300, 1: 300, 1.25: 300, 1.5: 350, 1.75: 350, 2: 400, 2.5: 400, 3: 500, 4: 800,
                     5: 1000, 6: 1200, 8: 500, 12: 600, 16: 650, 24: 700, 32: 900, 50: 1100, 64: 1200, 100: 1400,
                     150: 1600}

    beta_index = d_index // config['k_folds']
    beta = beta_rates[beta_index]
    epochs = epoch_granularity[beta]
    model_settings['beta'] = beta
    for k,v in model_settings.items():
        print(k, v)
    # maximum number of epochs to train to
    rounds = int(n_epochs_dict[beta] / epochs) + 1
    vae = VariationalAutoencoder(model_settings=model_settings)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.0))
    k = d_index % config['k_folds']
    training_input, validation_w_nan, validation_complete, val_na_ind = split_training_and_validation(config, k, data_missing_nan)
    for i in range(rounds):
        training_w_zeros = np.copy(training_input) # 667 obs
        validation_w_nan_cp = np.copy(validation_w_nan)
        history = vae.fit(x=training_w_zeros, y=training_w_zeros, epochs=epochs, batch_size=256)
        loss = int(round(history.history['loss'][-1] , 0))#  callbacks=[tensorboard_callback]
        results = evaluate_model(vae, validation_w_nan_cp, validation_complete, val_na_ind, scaler, config['recycles'], config['m'])
        completed_epochs = (i + 1) * epochs
        results['k'] = k
        save_results(results, completed_epochs, beta, results_path='LUADLUSC_beta_analysis.csv')
        remove_lock()
