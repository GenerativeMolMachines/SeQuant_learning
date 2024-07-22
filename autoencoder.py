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
    Dropout,
    MultiHeadAttention,
    Permute
)
from keras.models import Model
from keras.engine.keras_tensor import KerasTensor


def encoder(x: KerasTensor, height: int, width: int) -> tf.Tensor:
    # First block
    filters = height * 8
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=(filters // 2), dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]], axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = AveragePooling2D((1, 3))(x)
    x = Dropout(0.1)(x)
    width = width // 3

    # Second block
    filters = height * 8
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=(filters // 2), dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]], axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)
    width = width // 2

    # Third block
    filters = height * 4
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)
    width = width // 2

    # Fourth block
    filters = height * 2
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)
    width = width // 2

    # Fifth block
    filters = height
    x = Conv2D(filters, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=2, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)

    # Sixth block
    x = Conv2D(1, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    x = Dropout(0.1)(x)
    return x


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(units)(x)
    x = Dense(latent_dim, name='Latent')(x)
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((latent_dim, 1, 1))(x)
    return x


def decoder(x: KerasTensor, height: int, width: int) -> tf.Tensor:
    # First block
    x = Conv2DTranspose(1, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Second block
    filters = height
    width = width // 24
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=2, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    # Third block
    filters = height * 2
    width = width * 2
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    # Fourth block
    filters = height * 4
    width = width * 2
    x = Conv2DTranspose(filters, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=filters, dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]],
                               axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(
        attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    # Fifth block
    filters = height * 8
    width = width * 2
    x = Conv2DTranspose(height*8, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=(filters // 2), dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]], axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    # Sixth block
    filters = height * 8
    width = width * 3
    x = Conv2DTranspose(height*8, (1, 4), strides=(1, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x_permuted = Permute((2, 1, 3))(x)  # shape: (batch_size, width, height, filters)
    batch_size = tf.shape(x_permuted)[0]
    new_shape = tf.concat([[batch_size], [width], [height * filters]],
                          axis=0)  # shape: (batch_size, width, height * filters)
    x_reshaped = tf.reshape(x_permuted, new_shape)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=(filters // 2), dropout=0.1)(x_reshaped, x_reshaped)
    new_shape_back = tf.concat([[batch_size], [width], [height], [filters]], axis=0)  # shape: (batch_size, width, height, filters)
    attention_output_reshaped = tf.reshape(attention_output, new_shape_back)
    attention_output_permuted_back = Permute((2, 1, 3))(attention_output_reshaped)  # shape: (batch_size, height, width, filters)
    x = attention_output_permuted_back + x

    x = Activation('tanh')(x)
    return x


def autoencoder_model(height: int, width: int, channels: int, latent_dim: int, learning_rate: float) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x = encoder(x, height, width)
        x = latent_space(x, latent_dim)
        output = decoder(x, height)
        autoencoder = Model(inputs, output, name='Conv_Autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
    return autoencoder
