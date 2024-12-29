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
    Lambda
)
from keras.models import Model
from keras.engine.keras_tensor import KerasTensor


def get_filter_count(strategy: str, height: int, i: int, depth: int) -> int:
    if strategy == 'exponential':
        return 2 ** i
    elif strategy == 'linear':
        return 2 * (i + 1)
    elif strategy == 'height':
        return int(height * (i / 2))
    elif strategy == 'inverse_exponential':
        return 2 ** (depth - i)
    elif strategy == 'inverse_linear':
        if i != depth:
            filter_number = 2 * (depth - i)
        else:
            filter_number = 1
        return filter_number
    elif strategy == 'inverse_height':
        if i != depth:
            filter_number = int(height * ((depth - i) / 2))
        else:
            filter_number = 1
        return filter_number
    else:
        raise ValueError(f"Unsupported filter strategy: {strategy}")


def encoder(
        x: KerasTensor, height: int, width: int, depth: int,
        filter_strategy: str, noise: bool = False, noise_factor: float = 0.1
) -> tuple[KerasTensor, list]:

    if noise:
        stddev = tf.abs(x) * noise_factor
        noise_tensor = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
        x = tf.add(x, noise_tensor)

    n_values = []
    for i in range(1, depth+1):
        if i == 1:
            n = width // 2 ** (depth - 1)
        else:
            n = 2
        n_values.append(n)
        filters = get_filter_count(filter_strategy, height, i, depth)
        x = Conv2D(filters, (1, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D((1, n))(x)
        x = Dropout(0.1)(x)
    return x, n_values


def latent_space(x: tf.Tensor, latent_dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    filters = x.shape[3]
    x = Flatten()(x)

    # Mean value and standard deviation
    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    z = Lambda(sampling, name='Latent')([mu, log_var])
    z = Reshape((latent_dim, 1, filters))(z)

    return z, mu, log_var


def decoder(z: tf.Tensor, height: int, depth: int, n_values: list, filter_strategy: str) -> tf.Tensor:
    for i in range(1, depth+1):
        n = n_values[-i]
        count_list = list(number for number in range(1, 7))
        filters = get_filter_count(filter_strategy, height, count_list[-i], depth)
        z = Conv2DTranspose(filters, (1, 4), strides=(1, n), padding='same')(z)
        z = BatchNormalization()(z)
        if i == depth:
            z = Activation('tanh')(z)
        else:
            z = LeakyReLU(alpha=0.2)(z)
    return z


def vae_loss(original_inputs, outputs, mu, log_var):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(original_inputs, outputs))
    z_q = mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))
    z_p = tf.random.normal(tf.shape(mu))
    distances = tf.norm(tf.expand_dims(z_q, axis=1) - tf.expand_dims(z_p, axis=0), axis=-1)
    wasserstein = tf.reduce_mean(tf.reduce_min(distances, axis=1))

    return tf.reduce_mean(reconstruction_loss + wasserstein)


def autoencoder_model(
        height: int, width: int, depth: int, channels: int,
        latent_dim: int, learning_rate: float, filter_strategy: str,
        noise: bool = False, noise_factor: float = 0.1
) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x, n_values = encoder(x, height, width, depth, filter_strategy, noise, noise_factor)
        z, mu, log_var = latent_space(x, latent_dim)
        outputs = decoder(z, height, depth, n_values, filter_strategy)
        vae = Model(inputs, outputs, name='Variational_Autoencoder')
        vae.add_loss(vae_loss(inputs, outputs, mu, log_var))

        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return vae
