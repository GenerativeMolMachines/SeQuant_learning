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
from autoencoder import autoencoder_model

# timer
start_time = time.time()

# variables
max_len = 96
ratio_of_samples_to_use = 0.025
n_samples = 10000
num_seq = 10000
pad = -1
monomer_dict = {
    'A': 'CC(C(=O)O)N', 'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N',
    'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S', 'Q': 'C(CC(=O)N)C(C(=O)O)N',
    'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)CC(C(=O)O)N',
    'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N', 'K': 'C(CCN)CC(C(=O)O)N',
    'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]', 'water': 'O'
}
# hyperparameters
height = 46
width = 96
channels = 1
latent_dim = height
learning_rate = 1e-3
batch_size = 10
epochs = 100
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Processing labeled data (AMPs database)
labeled_data = pd.read_csv('data/AMP_ADAM2.txt', on_bad_lines='skip')
labeled_data = labeled_data.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)
labeled_data_seqs = labeled_data['SEQ'].to_list()

# Processing unlabeled data (Non-AMPs and CPPBase)
with open('data/Non-AMPs.txt') as f:
    file = f.readlines()
raw_seqs = file[1::2]
unlabeled_data = [s.replace("\n", "") for s in raw_seqs]

with open('data/natural_pep (cpp).txt') as f:
    file = f.readlines()
raw_seqs = file[1::2]
unlabeled_data_2 = [s.replace("\n", "") for s in raw_seqs]

# Process .fasta data (UniProt database)
'''
with open("/content/drive/MyDrive/SeQuant/UniProt/uniprot_sprot.txt","w") as f:
        for seq_record in SeqIO.parse("/content/drive/MyDrive/SeQuant/UniProt/uniprot_sprot.fasta", "fasta"):
                f.write(str(seq_record.seq) + "\n")
'''

with open('data/uniprot_sprot.txt') as f:
    raw_seqs_2 = f.readlines()
unlabeled_data_3 = [s.replace("\n", "") for s in raw_seqs_2]

# Process .fasta data (SPENCER database)
'''
with open("/content/drive/MyDrive/SeQuant/SPENCER/SPENCER_ORF_protein_sequence.txt","w") as f:
        for seq_record in SeqIO.parse("/content/drive/MyDrive/SeQuant/SPENCER/SPENCER_ORF_protein_sequence.fasta", "fasta"):
                f.write(str(seq_record.seq) + "\n")
'''

with open('data/SPENCER_ORF_protein_sequence.txt') as f:
    raw_seqs_2 = f.readlines()
unlabeled_data_4 = [s.replace("\n", "") for s in raw_seqs_2]

# Process .fasta data (HSPVdb database)
'''
with open("/content/drive/MyDrive/SeQuant/HSPVdb/hspvfullR58HET.txt","w") as f:
        for seq_record in SeqIO.parse("/content/drive/MyDrive/SeQuant/HSPVdb/hspvfullR58HET.fasta", "fasta"):
                f.write(str(seq_record.seq) + "\n")
'''

with open('data/hspvfullR58HET.txt') as f:
    raw_seqs_2 = f.readlines()
unlabeled_data_5 = [s.replace("\n", "") for s in raw_seqs_2]

# DBAASP database
dbaasp = pd.read_csv('data/peptides.csv')
dbaasp_2 = pd.read_csv('data/peptides_2.csv')
unlabeled_data_6 = list(
    dict.fromkeys(dbaasp['SEQUENCE'].astype('str').tolist() + dbaasp_2['SEQUENCE'].astype('str').tolist()))

# Merged data for CAE training
all_seqs = labeled_data_seqs + unlabeled_data + unlabeled_data_2 + unlabeled_data_3 + unlabeled_data_4 + unlabeled_data_5 + unlabeled_data_6

all_seqs = filter_sequences(all_seqs, monomer_dict)
all_seqs = [seq for seq in all_seqs if len(seq) <= max_len]
all_seqs = list(dict.fromkeys(all_seqs))
all_seqs_full = all_seqs

indices = len(all_seqs_full)
indices = list(range(indices))
idx_train = np.random.choice(indices, n_samples, replace=False)

all_seqs = list(itemgetter(*idx_train)(all_seqs_full))

#  use functions
descriptors_set = make_monomer_descriptors(monomer_dict)
encoded_sequences = encode_seqs(all_seqs, descriptors_set, max_len)
encoded_sequences = np.moveaxis(encoded_sequences, -1, 0)

## save encoded
# with open('/content/drive/MyDrive/SeQuant/encoded_sequences_' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
#         pad) + 'pad_alldescs_norm-1to1.pkl', 'wb') as f:
#     pickle.dump(encoded_sequences, f)
#
# with open('/content/drive/MyDrive/SeQuant/encoded_sequences_' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
#         pad) + 'pad_alldescs_norm-1to1.pkl', 'rb') as f:
#     enc_seqs = pickle.load(f)


# check if transformation is correct
assert np.all(
    seq_to_matrix(
        sequence=all_seqs[0],
        descriptors=descriptors_set,
        num=max_len
    ) == encoded_sequences[0, :, :]
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
checkpoint_filepath = 'checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Training
processed_data = preprocess_input(encoded_sequences)
X_train, X_test = train_test_split(processed_data, 0.7)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=epochs,
    validation_data=(X_test, X_test),
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback]
)

with open('trainHistoryDict/7_trainHistoryDict' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
        pad) + 'pad_alldescs_norm-1to1_batch' + str(batch_size) + '_lr' + str(learning_rate), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load model learning history
with open('trainHistoryDict/7_trainHistoryDict' + str(num_seq) + '_maxlen' + str(max_len) + '_' + str(
        pad) + 'pad_alldescs_norm-1to1_batch' + str(batch_size) + '_lr' + str(learning_rate), 'rb') as f:
    learning_history = pickle.load(f)

loss_hist = learning_history['loss']
val_loss_hist = learning_history['val_loss']
x_axis = range(1, len(loss_hist) + 1)

plt.plot(x_axis, loss_hist, color='r', label='loss')
plt.plot(x_axis, val_loss_hist, color='g', label='val_loss')
plt.yticks(np.arange(0, 0.5, 0.01))

plt.title("Autoencoder learning")

plt.legend()
plt.savefig('figures/7.png')

print("--- %s seconds ---" % (time.time() - start_time))
