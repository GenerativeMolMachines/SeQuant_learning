import os
import time
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from autoencoder_preset_tools import (
    create_dataset_from_batches,
    oversampling
)
from autoencoder_00 import autoencoder_model


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
learning_rate = 1e-3
batch_size = 32
epochs = 100
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
medium_test_df = pd.read_csv('data/medium_test_df.csv')
medium_train_df = pd.read_csv('data/medium_train_df.csv')

medium_train_data = list(medium_train_df['sequence'])
medium_test_data = list(medium_test_df['sequence'])

# Oversampling
medium_train_data = oversampling(sequences=medium_train_data, target_divisor=batch_size)
medium_test_data = oversampling(sequences=medium_test_data, target_divisor=batch_size)

np.random.shuffle(medium_train_data)
np.random.shuffle(medium_test_data)

# Batching
medium_train_batches = [medium_train_data[i:i + batch_size] for i in range(0, len(medium_train_data), batch_size)]
medium_test_batches = [medium_test_data[i:i + batch_size] for i in range(0, len(medium_test_data), batch_size)]

# tf.data.Dataset creation
medium_train_dataset = create_dataset_from_batches(batches=medium_train_batches, monomer_dict=monomer_dict, max_len=max_len)
medium_test_dataset = create_dataset_from_batches(batches=medium_test_batches, monomer_dict=monomer_dict, max_len=max_len)

for i in range(1, depth+1):
    print(f'Learning of the model with depth {i}')
    # model init
    autoencoder = autoencoder_model(
        height=height,
        width=width,
        depth=i,
        channels=channels,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )

    # set checkpoint
    checkpoint_filepath = f'checkpoint/checkpoint_0{i-1}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    start_time = time.time()

    # Training
    history = autoencoder.fit(
        medium_train_dataset,
        epochs=epochs,
        validation_data=medium_test_dataset,
        verbose=2,
        callbacks=[early_stop, model_checkpoint_callback]
    )

    with open(f'trainHistoryDict/0{i-1}.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # load model learning history
    with open(f'trainHistoryDict/0{i-1}.pkl', 'rb') as f:
        learning_history = pickle.load(f)

    print("--- %s seconds ---" % (time.time() - start_time))
    print()

print('Learning has been finished')
