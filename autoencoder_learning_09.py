import os
import time
import pickle
import numpy as np
import pandas as pd
import keras_tuner

import tensorflow as tf
from autoencoder_preset_tools import (
    create_dataset_from_batches,
    oversampling
)
from autoencoder_09 import autoencoder_model


# variables
max_len = 96
monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O','G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O', 'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O',
    'U': 'NC(C[Se])C(=O)O'
}
# hyperparameters
height = 46
width = 96
depth = 6
channels = 1
latent_dim = height
batch_size = 32
epochs = 50
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
small_test_df = pd.read_csv('data/small_test_df.csv')
small_train_df = pd.read_csv('data/small_train_df.csv')

small_train_data = list(small_train_df['sequence'])
small_test_data = list(small_test_df['sequence'])

# Oversampling
small_train_data = oversampling(sequences=small_train_data, target_divisor=batch_size)
small_test_data = oversampling(sequences=small_test_data, target_divisor=batch_size)

np.random.shuffle(small_train_data)
np.random.shuffle(small_test_data)

# Batching
small_train_batches = [small_train_data[i:i + batch_size] for i in range(0, len(small_train_data), batch_size)]
small_test_batches = [small_test_data[i:i + batch_size] for i in range(0, len(small_test_data), batch_size)]

# tf.data.Dataset creation
small_train_dataset = create_dataset_from_batches(batches=small_train_batches, monomer_dict=monomer_dict, max_len=max_len)
small_test_dataset = create_dataset_from_batches(batches=small_test_batches, monomer_dict=monomer_dict, max_len=max_len)


def build_model(hp):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, step=1e-3)
    filter_strategy = hp.Choice('filter_strategy', ['10_height', '8_height', '4_height', '12_height', '8_dupl_height'])
    activation_name = hp.Choice(
        'activation', ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'log_softmax']
    )
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.05)

    # model init
    autoencoder = autoencoder_model(
        height=height,
        width=width,
        depth=depth,
        channels=channels,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        filter_strategy=filter_strategy,
        activation_name=activation_name,
        dropout_rate=dropout_rate,

    )
    return autoencoder


# Init Keras Tuner
tuner = keras_tuner.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=epochs,
    factor=3,
    directory='tuning_results',
    project_name='wo_exp_act_lin_lr_depth_6'
)

# set checkpoint
checkpoint_filepath = f'checkpoint/checkpoint_09'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

start_time = time.time()

# Optimization
tuner.search(
    small_train_dataset,
    epochs=epochs,
    validation_data=small_test_dataset,
    callbacks=[early_stop, model_checkpoint_callback]
)

best_model = tuner.get_best_models(num_models=1)[0]

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best hyper params:
- Number of layers: {best_hps.get('num_layers')}
- Number of neurons in each layer: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}
- Learning rate: {best_hps.get('learning_rate')}
""")

# Training
history = best_model.fit(
    small_train_dataset,
    epochs=epochs,
    validation_data=small_test_dataset,
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback]
)

with open(f'trainHistoryDict/09.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load model learning history
with open(f'trainHistoryDict/09.pkl', 'rb') as f:
    learning_history = pickle.load(f)

print("--- %s seconds ---" % (time.time() - start_time))
print()

print('Optimizing has been finished')
