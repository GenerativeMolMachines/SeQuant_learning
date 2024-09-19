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


def encoder(x: KerasTensor, height: int) -> tf.Tensor:
    # First block
    filters = int(height * 0.5)
    x = Conv2D(filters, (1, 4), padding='same')(x) # input: (None, 46, 96, 1) output: (None, 46, 96, 23)
    x = BatchNormalization()(x) # output: (None, 46, 96, 23)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 96, 23)
    x = AveragePooling2D((1, 3))(x) # output: (None, 46, 32, 23)
    x = Dropout(0.1)(x) # output: (None, 46, 32, 23)

    # Second block
    filters = int(height * 1)
    x = Conv2D(filters, (1, 4), padding='same')(x) # output: (None, 46, 32, 46)
    x = BatchNormalization()(x) # output: (None, 46, 32, 46)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 32, 46)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 16, 46)
    x = Dropout(0.1)(x) # output: (None, 46, 16, 46)

    # Third block
    filters = int(height * 1.5)
    x = Conv2D(filters, (1, 4), padding='same')(x) # output: (None, 46, 16, 69)
    x = BatchNormalization()(x) # output: (None, 46, 16, 69)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 16, 69)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 8, 69)
    x = Dropout(0.1)(x) # output: (None, 46, 8, 69)

    # Fourth block
    filters = int(height * 2)
    x = Conv2D(filters, (1, 4), padding='same')(x) # output: (None, 46, 8, 92)
    x = BatchNormalization()(x) # output: (None, 46, 8, 92)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 8, 92)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 4, 92)
    x = Dropout(0.1)(x) # output: (None, 46, 4, 92)

    # Fifth block
    filters = int(height * 2.5)
    x = Conv2D(filters, (1, 4), padding='same')(x) # output: (None, 46, 4, 115)
    x = BatchNormalization()(x) # output: (None, 46, 4, 115)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 4, 115)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 2, 115)
    x = Dropout(0.1)(x) # output: (None, 46, 2, 115)

    # Sixth block
    filters = int(height * 3)
    x = Conv2D(filters, (1, 4), padding='same')(x) # output: (None, 46, 2, 138)
    x = BatchNormalization()(x) # output: (None, 46, 2, 138)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 2, 138)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 1, 138)
    return x


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x) # output: (None, 6348)
    units = x.shape[1]

    x = BatchNormalization()(x)
    x = Dense(units)(x) # output: (None, 6348)
    x = Dense(units // 4)(x) # output: (None, 1587)
    x = Dense(units // 16)(x) # output: (None, 396)
    x = Dense(units // 32)(x) # output: (None, 198)
    x = BatchNormalization()(x)

    x = Dense(latent_dim, name='Latent', activation='tanh')(x) # output: (None, 46)

    x = BatchNormalization()(x)
    x = Dense(units // 32)(x) # output: (None, 198)
    x = Dense(units // 16)(x) # output: (None, 396)
    x = Dense(units // 4)(x) # output: (None, 1587)
    x = Dense(units)(x) # output: (None, 6348)
    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x) # output: (None, 6348)
    x = Reshape((latent_dim, 1, -1))(x) # output: (None, 46, 1, 138)
    return x


def decoder(x: KerasTensor, height: int) -> tf.Tensor:
    # First block
    filters = int(height * 3)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Second block
    filters = int(height * 2.5)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Third block
    filters = int(height * 2)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fourth block
    filters = int(height * 1.5)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fifth block
    filters = int(height * 1)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Sixth block
    filters = int(height * 0.5)
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    return x


def autoencoder_model(height: int, width: int, channels: int, latent_dim: int, learning_rate: float) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x = encoder(x, height)
        x = latent_space(x, latent_dim)
        output = decoder(x, height)
        autoencoder = Model(inputs, output, name='Conv_Autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
    return autoencoder
