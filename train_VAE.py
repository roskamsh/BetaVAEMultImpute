from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import argparse
import json
import pickle
from betaVAE import VariationalAutoencoder
from bin.helper_functions import get_scaled_data

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='path to configuration json file')
parser.add_argument('--nextflow', type=bool, default=False, help = 'Whether you are running from a nextflow pipeline or not.')
parser.add_argument('--beta', type=float, default=1, help = 'Value of beta to train with.')

if __name__ == '__main__':
    args = parser.parse_args()
    configfile = args.config
    run_nextflow = args.nextflow
    beta = args.beta
    with open(configfile) as f:
        config = json.load(f)

    model_settings = \
        dict(
            latent_size=config["latent_size"],
            hidden_size_1=config["hidden_size_1"], 
            hidden_size_2=config["hidden_size_2"],  
            training_epochs = config["training_epochs"],
            batch_size = config["batch_size"],
            beta = beta,
            data_path = config["data_path"],
            corrupt_data_path = config["corrupt_data_path"],
            initial_imputation_strategy = config["initial_imputation_strategy"]
            )
    
    # Set up and scale dataframes
    data, data_missing = get_scaled_data(config["data_path"],config["corrupt_data_path"],
                                         initial_imputation_strategy=config["initial_imputation_strategy"],
                                         nextflow=run_nextflow)
    n_row = data.shape[1]
    model_settings['input_size']=n_row  # data input size
    
    vae = VariationalAutoencoder(model_settings=model_settings)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.0))
    history = vae.fit(x=data_missing, y=data_missing, epochs=config["training_epochs"], batch_size=config["batch_size"]) 
    print("current working directory:", os.getcwd())
    print("saving output to:", config["save_rootpath"])

    # Save model
    if (config["save_rootpath"]=="."):
        vae.save()
        with open('train_history.pickle', 'wb') as file_handle:
            pickle.dump(history.history, file_handle)
    else:
        vae.save(save_dir = config["save_rootpath"])
        with open(os.path.join(config["save_rootpath"],'train_history.pickle'), 'wb') as file_handle:
            pickle.dump(history.history, file_handle) 
