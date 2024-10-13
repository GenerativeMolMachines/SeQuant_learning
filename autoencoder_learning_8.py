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
from autoencoder_8 import autoencoder_model


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
channels = 1
latent_dim = height
learning_rate = 1e-3
batch_size = 32
epochs = 10
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
small_test_df = pd.read_csv('data/small_test_df.csv')
small_train_df = pd.read_csv('data/small_train_df.csv')
small_train_data = list(small_train_df['sequence'])
small_test_data = list(small_test_df['sequence'])

large_test_df = pd.read_csv('data/large_test_df.csv')
large_train_df = pd.read_csv('data/large_train_df.csv')
large_train_data = list(large_train_df['sequence'])
large_test_data = list(large_test_df['sequence'])

# Oversampling
small_train_data = oversampling(sequences=small_train_data, target_divisor=batch_size)
small_test_data = oversampling(sequences=small_test_data, target_divisor=batch_size)

np.random.shuffle(small_train_data)
np.random.shuffle(small_test_data)

large_train_data = oversampling(sequences=large_train_data, target_divisor=batch_size)
large_test_data = oversampling(sequences=large_test_data, target_divisor=batch_size)

np.random.shuffle(large_train_data)
np.random.shuffle(large_test_data)

# Batching
small_train_batches = [small_train_data[i:i + batch_size] for i in range(0, len(small_train_data), batch_size)]
small_test_batches = [small_test_data[i:i + batch_size] for i in range(0, len(small_test_data), batch_size)]

large_train_batches = [large_train_data[i:i + batch_size] for i in range(0, len(large_train_data), batch_size)]
large_test_batches = [large_test_data[i:i + batch_size] for i in range(0, len(large_test_data), batch_size)]

# tf.data.Dataset creation
small_train_dataset = create_dataset_from_batches(batches=small_train_batches, monomer_dict=monomer_dict, max_len=max_len)
small_test_dataset = create_dataset_from_batches(batches=small_test_batches, monomer_dict=monomer_dict, max_len=max_len)

large_train_dataset = create_dataset_from_batches(batches=large_train_batches, monomer_dict=monomer_dict, max_len=max_len)
large_test_dataset = create_dataset_from_batches(batches=large_test_batches, monomer_dict=monomer_dict, max_len=max_len)

# model init
autoencoder_small = autoencoder_model(
    height=height,
    width=width,
    channels=channels,
    latent_dim=latent_dim,
    learning_rate=learning_rate
)

autoencoder_large = autoencoder_model(
    height=height,
    width=width,
    channels=channels,
    latent_dim=latent_dim,
    learning_rate=learning_rate
)

# set checkpoint
checkpoint_filepath_small = 'checkpoint/checkpoint_8_small'
model_checkpoint_callback_small = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_small,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

checkpoint_filepath_large = 'checkpoint/checkpoint_8_large'
model_checkpoint_callback_large = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_large,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

start_time = time.time()

# Training
history_small = autoencoder_small.fit(
    small_train_dataset,
    epochs=epochs,
    validation_data=small_test_dataset,
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback_small]
)

history_large = autoencoder_large.fit(
    large_train_dataset,
    epochs=epochs,
    validation_data=large_test_dataset,
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback_large]
)

with open('trainHistoryDict/8_small.pkl', 'wb') as file_pi:
    pickle.dump(history_small.history, file_pi)

# load model learning history
with open('trainHistoryDict/8_small.pkl', 'rb') as f:
    learning_history = pickle.load(f)

with open('trainHistoryDict/8_large.pkl', 'wb') as file_pi:
    pickle.dump(history_large.history, file_pi)

# load model learning history
with open('trainHistoryDict/8_large.pkl', 'rb') as f:
    learning_history_large = pickle.load(f)

print("--- %s seconds ---" % (time.time() - start_time))
