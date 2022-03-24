import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.helper_functions import load_saved_model, apply_scaler


data_path =  "../data/data_complete.csv"
corrupt_data_path = "../data/LGGGBM_missing_10perc_trial1.csv"
vae = load_saved_model()
data = pd.read_csv(data_path).values
data = np.delete(data, np.s_[0:4], axis=1)
data_missing = pd.read_csv(corrupt_data_path).values

data, data_missing = apply_scaler(data, data_missing)

loss_per_cycle = vae.evaluate_on_true(data_missing, data, n_recycles=100, loss='RMSE')
plt.plot(loss_per_cycle)
plt.show()
