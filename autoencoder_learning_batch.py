import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from autoencoder_preset_tools import (
    create_dataset_from_batches,
    batch_creation
)
from autoencoder import autoencoder_model


# variables
max_len = 96
monomer_dict = {
    'A': 'CC(C(=O)O)N', 'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N',
    'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S', 'Q': 'C(CC(=O)N)C(C(=O)O)N',
    'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)CC(C(=O)O)N',
    'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N', 'K': 'C(CCN)CC(C(=O)O)N',
    'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]'
}
# hyperparameters
height = 46
width = 96
channels = 1
latent_dim = height
learning_rate = 1e-3
batch_size = 10000
epochs = 100
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced dataset
with open('data/test_seq_clean_str.pkl', 'rb') as f:
    test_data = pickle.load(f)

with open('data/train_seq_clean_str.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Batching
num_batches = len(train_data) // batch_size
train_batches = batch_creation(train_data, num_batches)
test_batches = batch_creation(test_data, num_batches)

# tf.data.Dataset creation
train_dataset = create_dataset_from_batches(batches=train_batches, monomer_dict=monomer_dict, max_len=max_len)
test_dataset = create_dataset_from_batches(batches=test_batches, monomer_dict=monomer_dict, max_len=max_len)

# model init
autoencoder = autoencoder_model(
    height=height,
    width=width,
    channels=channels,
    latent_dim=latent_dim,
    learning_rate=learning_rate
)

# set checkpoint
checkpoint_filepath = 'checkpoint/checkpoint_batching'
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
    train_dataset,
    train_dataset,
    epochs=epochs,
    validation_data=(test_dataset, test_dataset),
    verbose=2,
    callbacks=[early_stop, model_checkpoint_callback]
)

with open('trainHistoryDict/batching.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load model learning history
with open('trainHistoryDict/batching.pkl', 'rb') as f:
    learning_history = pickle.load(f)

print("--- %s seconds ---" % (time.time() - start_time))

loss_hist = learning_history['loss']
val_loss_hist = learning_history['val_loss']
x_axis = range(1, len(loss_hist) + 1)

plt.plot(x_axis, loss_hist, color='r', label='loss')
plt.plot(x_axis, val_loss_hist, color='g', label='val_loss')
plt.yticks(np.arange(0, 0.1, 0.005))

plt.title("Autoencoder learning")

plt.legend()
plt.savefig('figures/batching.png')

print("--- %s seconds ---" % (time.time() - start_time))
