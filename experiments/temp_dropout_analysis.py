import numpy as np
from betaVAEv2 import load_model_v2
from array_dropout_anlaysis import evaluate_model
from lib.helper_functions import get_scaled_data

model_dir = '../output/new_trained_model/epoch_1000/'
encoder_path = model_dir + 'encoder.keras'
decoder_path = model_dir + 'decoder.keras'
model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)

full_complete, full_w_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
missing_row_ind = np.where(np.isnan(full_w_nan).any(axis=1))[0]
missing_w_nans = full_w_nan[missing_row_ind]
missing_complete = full_complete[missing_row_ind]
na_ind = np.where(np.isnan(missing_w_nans))

results = evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler)
bp=True
