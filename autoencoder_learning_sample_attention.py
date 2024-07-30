import os
import time
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from autoencoder_preset_tools import (
    make_monomer_descriptors,
    seq_to_matrix,
    encode_seqs,
    preprocess_input,
)
from autoencoder import autoencoder_model


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
batch_size = 64
epochs = 100
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
protein_train = pd.read_csv('data/dna_rna/protein_train.csv')
protein_test = pd.read_csv('data/dna_rna/protein_test.csv')

# Use functions for protein datasets
descriptors_set = make_monomer_descriptors(monomer_dict)
#
protein_train_encoded_sequences = encode_seqs(protein_train['sequence'].tolist(), descriptors_set, max_len)
protein_train_encoded_sequences = np.moveaxis(protein_train_encoded_sequences, -1, 0)

protein_test_encoded_sequences = encode_seqs(protein_test['sequence'].tolist(), descriptors_set, max_len)
protein_test_encoded_sequences = np.moveaxis(protein_test_encoded_sequences, -1, 0)

# check if transformation is correct
protein_test_list = protein_test['sequence'].tolist()
assert np.all(
    seq_to_matrix(
        sequence=protein_test_list[0],
        descriptors=descriptors_set,
        num=max_len
    ) == protein_test_encoded_sequences[0, :, :]
)

X_train = protein_train_encoded_sequences
X_train = preprocess_input(X_train)
X_test = protein_test_encoded_sequences
X_test = preprocess_input(X_test)

# model init
autoencoder = autoencoder_model(
    height=height,
    width=width,
    channels=channels,
    latent_dim=latent_dim,
    learning_rate=learning_rate
)

# set checkpoint
checkpoint_filepath = 'checkpoint/checkpoint_sample_attention'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

start_time = time.time()

# Training
history = autoencoder.fit(
    X_train,
    X_train,
    epochs=epochs,
    validation_data=(X_test, X_test),
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback]
)

with open('trainHistoryDict/sample_attention.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load model learning history
with open('trainHistoryDict/sample_attention.pkl', 'rb') as f:
    learning_history = pickle.load(f)

print("--- %s seconds ---" % (time.time() - start_time))
