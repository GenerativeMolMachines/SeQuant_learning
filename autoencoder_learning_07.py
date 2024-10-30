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
from autoencoder_07 import autoencoder_model


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

# Filter strategies
filter_strategy = ['exponential', 'linear', 'height', 'inverse_exponential', 'inverse_linear', 'inverse_height']

# Preparation for embeddings saving
embeddings_dir = 'embeddings'
os.makedirs(embeddings_dir, exist_ok=True)

sampled_dataset = small_train_dataset.take(1000)
sampled_data = np.concatenate([x.numpy() for x, _ in sampled_dataset], axis=0)

# Architectures learning
for strategy in filter_strategy:
    # Clear previous model info
    tf.keras.backend.clear_session()

    print(f'Learning of the model with filter strategy: {strategy}')
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
    checkpoint_filepath = f'checkpoint/checkpoint_{strategy}'
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
        small_train_dataset,
        epochs=epochs,
        validation_data=small_test_dataset,
        verbose=2,
        callbacks=[early_stop, model_checkpoint_callback]
    )

    with open(f'trainHistoryDict/{strategy}.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # load model learning history
    with open(f'trainHistoryDict/{strategy}.pkl', 'rb') as f:
        learning_history = pickle.load(f)

    print("--- %s seconds ---" % (time.time() - start_time))
    print()

    # Embeddings calculation
    latent_layer_name = 'Latent'
    latent_model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(latent_layer_name).output)

    embeddings = latent_model.predict(sampled_data)

    embeddings_filepath = os.path.join(embeddings_dir, f'embeddings_{strategy}_1000.csv')
    pd.DataFrame(embeddings).to_csv(embeddings_filepath, index=False)

print('Learning has been finished')
