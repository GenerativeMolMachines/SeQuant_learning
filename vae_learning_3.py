import os
import pickle
import random
import time
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras import layers
from keras.layers import (
    AveragePooling2D,
    BatchNormalization
)

from autoencoder_preset_tools import data_processing
from vae import Sampling, VAE

os.environ["KERAS_BACKEND"] = "tensorflow"


start_time = time.time()
train_df = pd.read_csv('data/large_train_df.csv')
# test_df = pd.read_csv('data/small_test_df.csv')

random.seed(42)
# test_df = test_df.loc[random.sample(list(test_df.index),1000)]

train_data = list(train_df['sequence'])
print(len(train_data))
# test_data = list(test_df['sequence'])
# Oversampling
np.random.shuffle(train_data)
# np.random.shuffle(test_data)
# Data preprocessing
train_dataset = data_processing(batch_data=train_data, max_len=96)
# test_dataset = data_processing(batch_data=test_data, max_len=96)

gpus = tf.config.list_logical_devices('GPU')
print(gpus)
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    checkpoint_filepath = f'checkpoint/checkpoint_vae_zh' + '.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    latent_dim = 2
    encoder_inputs = keras.Input(shape=(46, 96, 1))
    x = layers.Conv2D(16, 3, activation="relu", strides=(1, 2), padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=(1, 4), padding="same")(x)
    x = layers.Conv2D(16, 3, activation="relu", strides=(2, 3), padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(286, activation="relu")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(46, name="Latent")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(286, activation="relu")(latent_inputs)
    x = BatchNormalization()(x)
    x = layers.Dense(1472, activation="relu")(x)
    x = layers.Reshape((23, 4, 16))(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=(2, 3), padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=(1, 4), padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=(1, 2), padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    print(decoder.summary())

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    input_shape = train_dataset.shape
    vae.build(input_shape)
    history = vae.fit(train_dataset, epochs=100, batch_size=128, callbacks=[early_stop, model_checkpoint_callback])
    with open(f'trainHistoryDict/vae_01.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

print("--- %s seconds ---" % (time.time() - start_time))

