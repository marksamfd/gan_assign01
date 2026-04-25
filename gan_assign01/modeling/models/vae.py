
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

from gan_assign01.config import IMG_SIZE, LATENT_DIM

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae():

    # Encoder
    encoder_inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,1))

    x = layers.Conv2D(32,3,activation='relu',padding='same')(encoder_inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z])

    # Decoder
    latent_inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(16*16*64, activation='relu')(latent_inputs)
    x = layers.Reshape((16,16,64))(x)

    x = layers.Conv2DTranspose(64,3,strides=2,padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(32,3,strides=2,padding='same',activation='relu')(x)

    decoder_outputs = layers.Conv2D(1,3,padding='same',activation='sigmoid')(x)

    decoder = Model(latent_inputs, decoder_outputs)

    # VAE
    class VAE(Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)

                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        binary_crossentropy(data, reconstruction),
                        axis=(1,2)
                    )
                )

                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1
                    )
                )

                total_loss = recon_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss
            }

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    return vae, encoder, decoder