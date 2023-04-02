import argparse
import json
import numpy as np
from bin.helper_functions import get_scaled_data
from cross_validation.run_cross_validation import evaluate_model, save_results, remove_lock
from betaVAE import VariationalAutoencoder
import tensorflow as tf


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cross_validation/cv_config.json', help='path to configuration json file')
    parser.add_argument('--latent', type=int, default=25, help='size of latent layer')
    parser.add_argument('--beta', type=int, default=2, help='beta value')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config['latent_size'] = args.latent
    config['beta'] = args.beta
    config["results_path"] = f"latent_{args.latent}_beta_{args.beta}_results.csv"
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
    data, data_missing_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True,
                                                     data_path=model_settings['data_path'],
                                                     corrupt_data_path=model_settings['corrupt_data_path'])
    n_row = data.shape[1]
    model_settings['input_size']=n_row
    epochs = 5
    config['m'] = 101
    for k,v in model_settings.items():
        print(k, v)
    # maximum number of epochs to train to
    rounds = 400
    vae = VariationalAutoencoder(model_settings=model_settings)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.0))
    training_input = np.nan_to_num(data_missing_nan)
    for i in range(rounds):
        training_w_zeros = np.copy(training_input) # 667 obs
        validation_w_nan_cp = np.copy(data_missing_nan)
        history = vae.fit(x=training_w_zeros, y=training_w_zeros, epochs=epochs, batch_size=256)
        loss = int(round(history.history['loss'][-1] , 0))#  callbacks=[tensorboard_callback]
        val_na_ind = np.where(np.isnan(validation_w_nan_cp))
        results = evaluate_model(vae, validation_w_nan_cp, data, val_na_ind, scaler, config['recycles'], config['m'])
        completed_epochs = (i + 1) * epochs
        results['k'] = k
        save_results(results, completed_epochs, config['beta'], results_path=config['results_path'], lock_path='latent_lock.txt')
        remove_lock(path='latent_lock.txt')