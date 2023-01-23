import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

try:
    with open("VAE_config.json") as f:
        config = json.load(f)
except:
    with open("../VAE_config.json") as f:
        config = json.load(f)
data_path = config["data_path"]
corrupt_data_path = config["corrupt_data_path"]
from sklearn.preprocessing import StandardScaler

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

def get_scaled_data(return_scaler=False, put_nans_back=False):
    running_dir = os.getcwd()
    for _ in range(3):
        if os.path.split(os.getcwd())[-1] == 'BetaVAEMImputation':
            break
        os.chdir('..')
    data = pd.read_csv(data_path).values
    data_missing = pd.read_csv(corrupt_data_path).values
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    del data_missing_complete
    data_missing[na_ind] = 0
    data_missing = sc.transform(data_missing)
    data = np.array(np.copy(data[:,4:]),dtype='float64')
    data = sc.transform(data)
    os.chdir(running_dir)
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

