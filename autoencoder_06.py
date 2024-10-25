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


def get_filter_count(strategy: str, height: int, i: int, depth: int) -> int:
    if strategy == '10_height':
        filter_sequence = [10, 5, 2.5, 1, 0.5]
    elif strategy == '8_height':
        filter_sequence = [8, 4, 2, 1, 0.5]
    elif strategy == '4_height':
        filter_sequence = [4, 4, 2, 1, 0.5]
    else:
        raise ValueError(f"Unsupported filter strategy: {strategy}")

    if i < 1 or i > depth:
        raise ValueError(f"Index i must be between 1 and {depth}, inclusive.")

    if i == depth:
        filter_number = 1
    else:
        filter_number = filter_sequence[i - 1] * height
    return int(filter_number)


def get_activation_layer(name: str):
    if name == 'leaky_relu':
        return LeakyReLU(alpha=0.2)
    elif name == 'relu':
        return Activation('relu')
    elif name == 'sigmoid':
        return Activation('sigmoid')
    elif name == 'softmax':
        return Activation('softmax')
    elif name == 'softplus':
        return Activation('softplus')
    elif name == 'tanh':
        return Activation('tanh')
    elif name == 'log_softmax':
        return Activation(tf.nn.log_softmax)
    elif name == 'exponential':
        return Activation('exponential')
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def encoder(
        x: KerasTensor, height: int, width: int, depth: int,
        filter_strategy: str, activation_name: str, dropout_rate: float
) -> tuple[tf.Tensor, list]:
    n_values = []
    for i in range(1, depth+1):
        if i == 1:
            n = width // 2 ** (depth - 1)
        else:
            n = 2
        n_values.append(n)
        filters = get_filter_count(filter_strategy, height, i, depth)
        x = Conv2D(filters=filters, kernel_size=(1, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = get_activation_layer(activation_name)(x)
        x = AveragePooling2D((1, n))(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    return x, n_values


def latent_space(x: KerasTensor, latent_dim: int, activation_name: str) -> tf.Tensor:
    filters = x.shape[3]
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(units)(x)
    x = Dense(latent_dim, name='Latent')(x)
    x = Dense(units)(x)
    x = get_activation_layer(activation_name)(x)
    x = Reshape((latent_dim, 1, filters))(x)
    return x


def decoder(
        x: KerasTensor, height: int, depth: int, n_values: list,
        filter_strategy: str, activation_name: str
) -> tf.Tensor:
    for i in range(1, depth+1):
        n = n_values[-i]
        count_list = list(number for number in range(1, 7))
        filters = get_filter_count(filter_strategy, height, count_list[-i], depth)
        x = Conv2DTranspose(filters=filters, kernel_size=(1, 4), strides=(1, n), padding='same')(x)
        x = BatchNormalization()(x)
        if i == depth:
            x = Activation('tanh')(x)
        else:
            x = get_activation_layer(activation_name)(x)
    return x


def autoencoder_model(
        height: int, width: int, depth: int, channels: int,
        latent_dim: int, learning_rate: float, filter_strategy: str,
        activation_name: str, dropout_rate: float) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x, n_values = encoder(x, height, width, depth, filter_strategy, activation_name, dropout_rate)
        x = latent_space(x, latent_dim, activation_name)
        output = decoder(x, height, depth, n_values, filter_strategy, activation_name)
        autoencoder = Model(inputs, output, name='Conv_Autoencoder')
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
    return autoencoder
