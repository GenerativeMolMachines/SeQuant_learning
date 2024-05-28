import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

import tensorflow as tf
from autoencoder_preset_tools import (
    make_monomer_descriptors,
    seq_to_matrix,
    encode_seqs,
    preprocess_input,
    train_test_split,
    filter_sequences
)
from autoencoder_optimal import autoencoder_model

# timer
start_time = time.time()

# variables
max_len = 96
ratio_of_samples_to_use = 0.025
n_samples = 100000
num_seq = 100000
pad = -1
monomer_dict = {
    'dA': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)C[C@@H]3O',  # DNA
    'dT': r'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O',
    'dG': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n1cnc2c1NC(=N/C2=O)\N)C[C@@H]3O',
    'dC': r'C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)CO[P@@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O',
    'rA': r'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N',  # RNA
    'rU': r'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O',
    'rG': r'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N',
    'rC': r'c1cn(c(=O)nc1N)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O',
    'A': 'CC(C(=O)O)N',  # protein
    'R': 'C(CC(C(=O)O)N)CN=C(N)N',
    'N': 'C(C(C(=O)O)N)C(=O)N',
    'D': 'C(C(C(=O)O)N)C(=O)O',
    'C': 'C(C(C(=O)O)N)S',
    'Q': 'C(CC(=O)N)C(C(=O)O)N',
    'E': 'C(CC(=O)O)C(C(=O)O)N',
    'G': 'C(C(=O)O)N',
    'H': 'C1=C(NC=N1)CC(C(=O)O)N',
    'I': 'CCC(C)C(C(=O)O)N',
    'L': 'CC(C)CC(C(=O)O)N',
    'K': 'C(CCN)CC(C(=O)O)N',
    'M': 'CSCCC(C(=O)O)N',
    'F': 'C1=CC=C(C=C1)CC(C(=O)O)N',
    'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O',
    'T': 'CC(C(C(=O)O)N)O',
    'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O',
    'V': 'CC(C)C(C(=O)O)N',
    'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]'
}
# hyperparameters
height = 43
width = 96
channels = 1
latent_dim = height
learning_rate = 1e-3
batch_size = 10
epochs = 1000
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading balanced datasets
dna_train = list(pd.read_csv('data/dna_rna/dna_train.csv'))
dna_test = list(pd.read_csv('data/dna_rna/dna_test.csv'))

rna_train = list(pd.read_csv('data/dna_rna/rna_train.csv'))
rna_test = list(pd.read_csv('data/dna_rna/rna_test.csv'))

protein_train = list(pd.read_csv('data/dna_rna/protein_train.csv'))
protein_test = list(pd.read_csv('data/dna_rna/protein_test.csv'))

# Use functions for DNA datasets
descriptors_set = make_monomer_descriptors(monomer_dict)

dna_train_encoded_sequences = encode_seqs(dna_train, descriptors_set, max_len, polymer_type='DNA')
dna_train_encoded_sequences = np.moveaxis(dna_train_encoded_sequences, -1, 0)

dna_test_encoded_sequences = encode_seqs(dna_test, descriptors_set, max_len, polymer_type='DNA')
dna_test_encoded_sequences = np.moveaxis(dna_test_encoded_sequences, -1, 0)

# Use functions for RNA datasets
rna_train_encoded_sequences = encode_seqs(rna_train, descriptors_set, max_len, polymer_type='RNA')
rna_train_encoded_sequences = np.moveaxis(rna_train_encoded_sequences, -1, 0)

rna_test_encoded_sequences = encode_seqs(rna_test, descriptors_set, max_len, polymer_type='RNA')
rna_test_encoded_sequences = np.moveaxis(rna_test_encoded_sequences, -1, 0)

# Use functions for protein datasets
protein_train_encoded_sequences = encode_seqs(protein_train, descriptors_set, max_len, polymer_type='protein')
protein_train_encoded_sequences = np.moveaxis(protein_train_encoded_sequences, -1, 0)

protein_test_encoded_sequences = encode_seqs(protein_test, descriptors_set, max_len, polymer_type='protein')
protein_test_encoded_sequences = np.moveaxis(protein_test_encoded_sequences, -1, 0)

# check if transformation is correct
assert np.all(
    seq_to_matrix(
        sequence=protein_test[0],
        descriptors=descriptors_set,
        num=max_len
    ) == protein_test_encoded_sequences[0, :, :]
)

# model init
autoencoder = autoencoder_model(
    height=height,
    width=width,
    channels=channels,
    latent_dim=latent_dim,
    learning_rate=learning_rate
)

# set checkpoint
checkpoint_filepath = 'checkpoint/checkpoint_all_polymers'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Training
X_train = np.hstack([dna_train_encoded_sequences, rna_train_encoded_sequences, protein_train_encoded_sequences])
X_train = preprocess_input(X_train)

X_test = np.hstack([dna_test_encoded_sequences, rna_test_encoded_sequences, protein_test_encoded_sequences])
X_test = preprocess_input(X_test)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=epochs,
    validation_data=(X_test, X_test),
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback]
)

with open('trainHistoryDict/all_polymers_trainHistoryDict' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
        pad) + 'pad_alldescs_norm-1to1_batch' + str(batch_size) + '_lr' + str(learning_rate), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load model learning history
with open('trainHistoryDict/all_polymers_trainHistoryDict' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
        pad) + 'pad_alldescs_norm-1to1_batch' + str(batch_size) + '_lr' + str(learning_rate), 'rb') as f:
    learning_history = pickle.load(f)

loss_hist = learning_history['loss']
val_loss_hist = learning_history['val_loss']
x_axis = range(1, len(loss_hist) + 1)

plt.plot(x_axis, loss_hist, color='r', label='loss')
plt.plot(x_axis, val_loss_hist, color='g', label='val_loss')
plt.yticks(np.arange(0, 0.1, 0.005))

plt.title("Autoencoder learning")

plt.legend()
plt.savefig('figures/all_polymers.png')

print("--- %s seconds ---" % (time.time() - start_time))
