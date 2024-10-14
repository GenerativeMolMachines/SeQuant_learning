import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    AveragePooling2D,
    BatchNormalization,
    Flatten,
    Reshape,
    Conv2DTranspose,
    LeakyReLU,
    Activation,
    Dropout
)
from keras.models import Model
from keras.engine.keras_tensor import KerasTensor


def encoder(x: KerasTensor, height: int, width: int, depth: int) -> tuple[tf.Tensor, list]:
    n_values = []
    for i in range(1, depth+1):
        if i == 1:
            n = width // 2 ** (depth - 1)
        else:
            n = 2
        n_values.append(n)
        x = Conv2D(height, (1, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D((1, n))(x)
        x = Dropout(0.1)(x)
    return x, n_values


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(units)(x)
    x = Dense(latent_dim, name='Latent')(x)
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((latent_dim, 1, 1))(x)
    return x


def decoder(x: KerasTensor, height: int, depth: int, n_values: list) -> tf.Tensor:
    for i in range(1, depth+1):
        n = n_values[-i]
        x = Conv2DTranspose(height, (1, 4), strides=(1, n), padding='same')(x)
        x = BatchNormalization()(x)
        if i == depth:
            x = Activation('tanh')(x)
        else:
            x = LeakyReLU(alpha=0.2)(x)
    return x


def autoencoder_model(height: int, width: int, depth: int, channels: int, latent_dim: int, learning_rate: float) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x, n_values = encoder(x, height, width, depth)
        x = latent_space(x, latent_dim)
        output = decoder(x, height, depth, n_values)
        autoencoder = Model(inputs, output, name='Conv_Autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
    return autoencoder
