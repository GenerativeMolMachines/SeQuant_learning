import os
import pickle
import random
import time
import numpy as np
import pandas as pd

import keras
import tensorflow as tf


from autoencoder_preset_tools import data_processing
from vae import VAE

os.environ["KERAS_BACKEND"] = "tensorflow"


start_time = time.time()
train_df = pd.read_csv('data/small_train_df.csv')
# test_df = pd.read_csv('data/small_test_df.csv')

random.seed(42)
train_df = train_df.loc[random.sample(list(train_df.index),30000)]
# test_df = test_df.loc[random.sample(list(test_df.index),1000)]

train_data = list(train_df['sequence'])
# test_data = list(test_df['sequence'])
# Oversampling
np.random.shuffle(train_data)
# np.random.shuffle(test_data)
# Data preprocessing
train_dataset = data_processing(batch_data=train_data, max_len=96)
# test_dataset = data_processing(batch_data=test_data, max_len=96)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    checkpoint_filepath = f'checkpoint/checkpoint_vae_zh' + '.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='total_loss',
        mode='min',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience=3)

    vae = VAE()
    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit(train_dataset, epochs=100, batch_size=128, callbacks=[early_stop, model_checkpoint_callback])
    with open(f'trainHistoryDict/vae_01.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

print("--- %s seconds ---" % (time.time() - start_time))
