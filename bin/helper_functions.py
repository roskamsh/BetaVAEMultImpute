import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

import tensorflow_probability as tfp

param_imputation = {
        'strategy': 'mean',  # for simple imputer  (mean or median)
        'n_neighbors': 5,    # for knn imputer
        'max_iter': 20,      # for iterative imputer
        'tol': 1e-3          # for iterative imputer
    }

def evaluate_coverage_quantile(multi_imputes, data, data_missing, scaler):
    na_ind = np.where(np.isnan(data_missing))
    true_values = data[na_ind]
    low_q80 = np.percentile(multi_imputes, 10, axis=0)
    up_q80 = np.percentile( multi_imputes,90, axis=0)
    low_q90 = np.percentile( multi_imputes,5, axis=0)
    up_q90 = np.percentile( multi_imputes,95, axis=0)
    low_q95 = np.percentile(multi_imputes,2.5,  axis=0)
    up_q95 = np.percentile(multi_imputes,97.5,  axis=0)
    low_q99 = np.percentile( multi_imputes,0.5, axis=0)
    up_q99 = np.percentile( multi_imputes,99.5, axis=0)
    results = {
        'prop_80q': np.array([low_q80[i] < true_values[i] < up_q80[i] for i in range(len(true_values))]).mean(),
        'prop_90q': np.array([low_q90[i] < true_values[i] < up_q90[i] for i in range(len(true_values))]).mean(),
        'prop_95q': np.array([low_q95[i] < true_values[i] < up_q95[i] for i in range(len(true_values))]).mean(),
        'prop_99q': np.array([low_q99[i] < true_values[i] < up_q99[i] for i in range(len(true_values))]).mean(),
    }
    return results

def evaluate_coverage(multi_imputes, data, data_missing, scaler):
    assert data_missing.shape == data.shape
    na_ind = np.where(np.isnan(data_missing))
    means = np.mean(multi_imputes, axis=0)
    unscaled_st_devs = np.std(multi_imputes, axis=0)
    unscaled_differences = np.abs(data[na_ind] - means)
    n_deviations = unscaled_differences / unscaled_st_devs
    ci_80 = 1.282
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_80 = sum(n_deviations < ci_80) / len(n_deviations)
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    results = {
        'prop_80': prop_80,
        'prop_90': prop_90,
        'prop_95': prop_95,
        'prop_99': prop_99
    }
    for k, v in results.items():
        print(k,':', v)
    data = scaler.inverse_transform(data)
    data_missing[na_ind] = means
    data_missing = scaler.inverse_transform(data_missing)
    differences = np.abs(data[na_ind] - data_missing[na_ind])
    MAE = np.mean(differences)
    results['multi_mae'] = MAE
    print('average absolute error:', MAE)
    return results

def impute_nas_with_zeros(data_missing):
    data_imputed = data_missing.copy()
    na_ind = np.where(np.isnan(data_imputed))
    data_imputed[na_ind] = 0
    return data_imputed

def impute_nas_with_iterative_imputer(data_missing, type_imputer, params):
    if isinstance(data_missing, np.ndarray):
        dataframe_with_nans = pd.DataFrame(data_missing)
    else:
        dataframe_with_nans = data_missing.copy()
    if type_imputer == "simple":
        imp = SimpleImputer(missing_values=np.nan, strategy=params['strategy'])
    elif type_imputer == "knn":
        imp = KNNImputer(missing_values=np.nan, n_neighbors=params['n_neighbors'])
    elif type_imputer == "iterative_bayesridge":  # regularized linear regression
        imp = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,
                               max_iter=params['max_iter'], tol=params['tol'])
    elif type_imputer == "iterative_randomforest":  # Forests of randomized trees regression
        imp = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan,
                               max_iter=params['max_iter'], tol=params['tol'],verbose=1)
    else:
        raise ValueError(f"Invalid type of imputer chosen: {type_imputer}. Choose from 'simple', 'knn', 'iterative'.")
    data_imputed = imp.fit_transform(dataframe_with_nans)

    if isinstance(data_missing, np.ndarray):
        return data_imputed
    elif isinstance(data_missing, pd.DataFrame):
        return data_imputed.to_numpy()
    return data_imputed
    
def perform_initial_imputation(data_missing, type_imputer, params=param_imputation):
    """
    Here we run initial imputation at missing value indicies. 
    This can be done either by using the IterativeImputer, specifying options "knn", "iterative_bayesridge" or "iterative_randomforest".
    However, this method is very slow / not feasible for large dimensions, so we recommend imputing with zeros to start, for z-scored data.
    """

    if type_imputer == "zero":
        data_imputed = impute_nas_with_zeros(data_missing)
    elif type_imputer in ["simple","knn","iterative_bayesridge","iterative_randomforest"]:
        data_imputed = impute_nas_with_iterative_imputer(data_missing, type_imputer, params)
    else:
        raise ValueError(f"Invalid type of imputer chosen: {type_imputer}. Choose from 'simple', 'knn', 'iterative', or 'zero'.")

    return data_imputed

def get_scaled_data(data_path, corrupt_data_path, initial_imputation_strategy, return_scaler=False, put_nans_back=False, nextflow=False):
    data_fn = os.path.basename(data_path)
    corrupt_data_fn = os.path.basename(corrupt_data_path) 
    # If running in nextflow, use the data & corrupt data in cwd
    if nextflow:
        data = pd.read_csv(os.path.join(os.getcwd(),data_fn)).values
        data_missing = pd.read_csv(os.path.join(os.getcwd(),corrupt_data_fn)).values 
    else:
        data = pd.read_csv(data_path).values 
        data_missing = pd.read_csv(corrupt_data_path).values
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    del data_missing_complete
    data_missing = perform_initial_imputation(data_missing, type_imputer = initial_imputation_strategy)
    data_missing = sc.transform(data_missing)
    data = np.array(np.copy(data[:,4:]),dtype='float64')
    data = sc.transform(data)
    if put_nans_back:
        data_missing[na_ind] = np.nan
    if return_scaler:
        return data, data_missing, sc
    else:
        return data, data_missing

def apply_scaler(data, data_missing, return_scaler=False):
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    data_missing[na_ind] = 0
    # Scale the testing data with model's trianing data mean and variance
    data_missing = sc.transform(data_missing)
    data_missing[na_ind] = np.nan
    del data_missing_complete
    data = sc.transform(data)
    if return_scaler:
        return data, data_missing, sc
    else:
        return data, data_missing

def mc_wrt_p(data_complete, num_samples_mc, model, z_prior, want_log=True, observed_indices_mask=None):
    N = data_complete.shape[0]
    beta = model.beta
    
    zmc = z_prior.sample((num_samples_mc,)).numpy()
    
    logpy_byobs = []
    for i in range(N):
        data_complete_i = data_complete[i]
        zi = zmc[:,i,:]
        x_hat_mean, x_hat_log_sigma_sq = model.decoder.predict(zi)
        x_hat_sigma = np.exp(0.5 * x_hat_log_sigma_sq)
        X_hat_distribution = tfp.distributions.Normal(loc=x_hat_mean, scale=np.sqrt(beta)*x_hat_sigma) # size [num_samples_mcmc, num_features]
        if want_log:
            # Here logpy is an array with the logp for observation i and each sample across num_samples_mcmc
            # Will be size [num_samples_mc]
            if observed_indices_mask is not None:
                obs_mask_i = observed_indices_mask[i,:].copy()
                logpy = tf.reduce_sum(X_hat_distribution.log_prob(data_complete_i).numpy() * obs_mask_i, axis=1).numpy()  
            else:
                logpy = tf.reduce_sum(X_hat_distribution.log_prob(data_complete_i).numpy(), axis=1).numpy() 
            c = np.max(logpy)
            internal_mean = np.mean(np.exp(logpy - c), axis=0)
            result = np.log(internal_mean) + c
        else:
            logpy = tf.reduce_sum(X_hat_distribution.prob(data_complete_i).numpy(), axis=1).numpy()
            result = np.mean(logpy, axis=0)
        logpy_byobs.append(result)
    return np.array(logpy_byobs)

def log_lik_ymis_given_obs_mcmc_p(data, data_corrupt, model, num_samples_mc=500):
    latent_dim = model.latent_dim
    missing_row_ind = np.where(np.isnan(data_corrupt).any(axis=1))
    data_corrupt_at_missing_samples = data_corrupt[missing_row_ind[0],:]
    data_complete_at_missing_samples = data[missing_row_ind[0],:]
    compl_ind = np.where(np.isfinite(data_corrupt_at_missing_samples))
    observed_indices_mask = np.zeros(data_complete_at_missing_samples.shape)
    observed_indices_mask[compl_ind] = 1
    z_prior = tfp.distributions.Normal(
            loc=np.zeros([data_complete_at_missing_samples.shape[0], latent_dim]), 
            scale=np.ones([data_complete_at_missing_samples.shape[0], latent_dim])
    )

    # log_p_y is a list of length N_samp, with the approximation of logp(y_true) from mc
    log_p_y = mc_wrt_p(data_complete_at_missing_samples, num_samples_mc=num_samples_mc, model=model, 
                       z_prior=z_prior, observed_indices_mask=None, want_log=True)
    log_p_yobs = mc_wrt_p(data_complete_at_missing_samples, num_samples_mc=num_samples_mc, model=model, 
                       z_prior=z_prior, observed_indices_mask=observed_indices_mask, want_log=True)
    
    log_p_y_mis_given_obs = log_p_y - log_p_yobs

    return log_p_y_mis_given_obs

class DataMissingMaker: # TODO remove this unused class
    def __init__(self, complete_only, prop_miss_rows=1, prop_miss_col=0.1):
        self.data = complete_only
        self.n_col = self.data.shape[1]
        self.prop_miss_rows = prop_miss_rows
        self.prop_miss_col = prop_miss_col
        self.n_rows_to_null = int(len(complete_only) * prop_miss_rows)


    def get_random_col_selection(self):
        n_cols_to_null = np.random.binomial(n=self.n_col, p=self.prop_miss_col)
        return np.random.choice(range(self.n_col), n_cols_to_null, replace=False)

    def generate_missing_data(self):
        random_rows = np.random.choice(range(len(self.data)), self.n_rows_to_null, replace=False)
        null_col_indexes = [self.get_random_col_selection() for _ in range(self.n_rows_to_null)]
        null_row_indexes = [np.repeat(row, repeats=len(null_col_indexes[i])) for i, row in enumerate(random_rows)]
        null_col_indexes = np.array([inner[j] for inner in null_col_indexes for j in range(len(inner))]) # flatten the nested arrays
        null_row_indexes = np.array([inner[j] for inner in null_row_indexes for j in range(len(inner))]) # flatten the nested arrays
        new_masked_x = np.copy(self.data)
        new_masked_x[null_row_indexes, null_col_indexes] = np.nan
        return new_masked_x

