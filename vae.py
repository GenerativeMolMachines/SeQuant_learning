import tensorflow as tf
import os
import keras
from keras import ops
from keras import layers
from keras.layers import (
    AveragePooling2D,
    BatchNormalization
)
os.environ["KERAS_BACKEND"] = "tensorflow"


class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = 2
        self.encoder = self.make_encoder(self.latent_dim)
        self.decoder = self.make_decoder(self.latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @staticmethod
    def make_encoder(latent_dim: int):
        encoder_inputs = keras.Input(shape=(46, 96, 1))
        x = layers.Conv2D(16, 3, activation="leaky_relu", strides=1, padding="same")(encoder_inputs)
        x = AveragePooling2D((1, 2))(x)
        x = layers.Conv2D(32, 2, activation="leaky_relu", strides=1, padding="same")(x)
        x = AveragePooling2D((1, 2))(x)
        x = layers.Conv2D(64, 3, activation="leaky_relu", strides=2, padding="same")(x)
        x = AveragePooling2D((1, 4))(x)

        x = layers.Flatten()(x)
        x = BatchNormalization()(x)
        x = layers.Dense(46, name="Latent")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    @staticmethod
    def make_decoder(latent_dim: int):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(4416, activation="relu")(latent_inputs)
        x = layers.Reshape((23, 3, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="leaky_relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="leaky_relu", strides=(1, 4), padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, activation="leaky_relu", strides=(1, 4), padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder
