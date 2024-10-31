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
    filters = height * 1
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # input: (None, 46, 96, 1) output: (None, 46, 96, 46)
    x = BatchNormalization()(x) # output: (None, 46, 96, 46)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 96, 46)
    x = AveragePooling2D((1, 3))(x) # output: (None, 46, 32, 46)
    x = Dropout(0.1)(x) # output: (None, 46, 32, 46)

    # Second block
    filters = height * 2
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # output: (None, 46, 32, 92)
    x = BatchNormalization()(x) # output: (None, 46, 32, 92)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 32, 92)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 16, 92)
    x = Dropout(0.1)(x) # output: (None, 46, 16, 92)

    # Third block
    filters = height * 4
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # output: (None, 46, 16, 184)
    x = BatchNormalization()(x) # output: (None, 46, 16, 184)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 16, 184)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 8, 184)
    x = Dropout(0.1)(x) # output: (None, 46, 8, 184)

    # Fourth block
    filters = height * 6
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # output: (None, 46, 8, 276)
    x = BatchNormalization()(x) # output: (None, 46, 8, 276)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 8, 276)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 4, 276)
    x = Dropout(0.1)(x) # output: (None, 46, 4, 276)

    # Fifth block
    filters = height * 8
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # output: (None, 46, 4, 368)
    x = BatchNormalization()(x) # output: (None, 46, 4, 368)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 4, 368)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 2, 368)
    x = Dropout(0.1)(x) # output: (None, 46, 2, 368)

    # Sixth block
    filters = height * 10
    x = Conv2D(filters, (1, 4), padding='same', kernel_regularizer='l1_l2')(x) # output: (None, 46, 2, 460)
    x = BatchNormalization()(x) # output: (None, 46, 2, 460)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 46, 2, 460)
    x = AveragePooling2D((1, 2))(x) # output: (None, 46, 1, 460)
    x = Dropout(0.1)(x) # output: (None, 46, 1, 460)
    return x


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x) # output: (None, 21160)
    units = x.shape[1]
    x = Dense(units, kernel_regularizer='l1_l2', activity_regularizer='l1_l2')(x)  # output: (None, 21160)
    x = Dense(latent_dim, kernel_regularizer='l1_l2', activity_regularizer='l1_l2', name='Latent')(x) # output: (None, 46)
    x = Dense(units)(x) # output: (None, 21160)
    x = LeakyReLU(alpha=0.2)(x) # output: (None, 21160)
    x = Reshape((latent_dim, 1, -1))(x) # output: (None, 46, 1, 460)
    return x


def decoder(x: KerasTensor, height: int) -> tf.Tensor:
    # First block
    filters = height * 10
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same', kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Second block
    filters = height * 8
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same', kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Third block
    filters = height * 6
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same', kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fourth block
    filters = height * 4
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same', kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fifth block
    filters = height * 2
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same', kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Sixth block
    filters = height * 1
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 3), padding='same', kernel_regularizer='l1_l2')(x)
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
