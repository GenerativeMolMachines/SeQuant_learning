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
from vae_010 import autoencoder_model


# variables
max_len = 96
monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
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
epochs = 30
tf.keras.backend.clear_session()
tf.random.set_seed(42)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
"""
small_test_df = pd.read_csv('data/small_test_df.csv')
small_train_df = pd.read_csv('data/small_train_df.csv')

medium_test_df = pd.read_csv('data/medium_test_df.csv')
medium_train_df = pd.read_csv('data/medium_train_df.csv')

large_test_df = pd.read_csv('data/large_test_df.csv')
large_train_df = pd.read_csv('data/large_train_df.csv')

train_df = pd.concat([small_train_df, medium_train_df, large_train_df], axis=0)
test_df = pd.concat([small_test_df, medium_test_df, large_test_df], axis=0)
"""
train_df = pd.read_csv('data/small_train_df.csv')
test_df = pd.read_csv('data/small_test_df.csv')

train_data = list(train_df['sequence'])
test_data = list(test_df['sequence'])

# Oversampling
train_data = oversampling(sequences=train_data, target_divisor=batch_size)
test_data = oversampling(sequences=test_data, target_divisor=batch_size)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Batching
train_batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
test_batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

# tf.data.Dataset creation
train_dataset = create_dataset_from_batches(batches=train_batches, monomer_dict=monomer_dict, max_len=max_len)
test_dataset = create_dataset_from_batches(batches=test_batches, monomer_dict=monomer_dict, max_len=max_len)

# Filter strategies
filter_strategy = ['inverse_height']
"""
# Preparation for embeddings saving
embeddings_dir = 'embeddings'
os.makedirs(embeddings_dir, exist_ok=True)

sampled_dataset = train_dataset.take(1000)
sampled_data = np.concatenate([x.numpy() for x, _ in sampled_dataset], axis=0)
"""
# Architectures learning
for strategy in filter_strategy:
    # Clear previous model info
    tf.keras.backend.clear_session()

    print(f'Learning of the model with filter strategy: {strategy}, depth = {depth}')
    # model init
    autoencoder = autoencoder_model(
        height=height,
        width=width,
        depth=depth,
        channels=channels,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        filter_strategy=strategy
    )

    # set checkpoint
    checkpoint_filepath = f'checkpoint/checkpoint_vae_004'
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
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=2,
        callbacks=[early_stop, model_checkpoint_callback]
    )

    with open(f'trainHistoryDict/vae_004.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # load model learning history
    with open(f'trainHistoryDict/vae_004.pkl', 'rb') as f:
        learning_history = pickle.load(f)

    print("--- %s seconds ---" % (time.time() - start_time))
    print()

    print(autoencoder.summary())
"""
    # Embeddings calculation
    latent_layer_name = 'Latent'
    latent_model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(latent_layer_name).output)

    embeddings = latent_model.predict(sampled_data)

    embeddings_filepath = os.path.join(embeddings_dir, f'vae_embeddings_{strategy}_1000.csv')
    pd.DataFrame(embeddings).to_csv(embeddings_filepath, index=False)
"""
print('Learning has been finished')
