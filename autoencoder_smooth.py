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
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 3))(x)
    x = Dropout(0.1)(x)

    # Second block
    filters = int(height * 1)
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)

    # Third block
    filters = int(height * 1.5)
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)

    # Fourth block
    filters = int(height * 2)
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)

    # Fifth block
    filters = int(height * 2.5)
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)

    # Sixth block
    filters = int(height * 3)
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    return x


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x)
    units = x.shape[1]

    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 2)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 4)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 8)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 16)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 32)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 64)(x)
    x = BatchNormalization()(x)

    x = Dense(latent_dim, name='Latent', activation='tanh')(x)

    x = Dense(units // 64)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 32)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 16)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 8)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 4)(x)
    x = BatchNormalization()(x)
    x = Dense(units // 2)(x)
    x = BatchNormalization()(x)
    x = Dense(units)(x)
    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((latent_dim, 1, -1))(x)
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
