
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

from gan_assign01.config import IMG_SIZE, LATENT_DIM

def build_autoencoder():

    inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,1))

    x = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    latent = layers.Dense(LATENT_DIM)(x)

    x = layers.Dense(16*16*64, activation='relu')(latent)
    x = layers.Reshape((16,16,64))(x)

    x = layers.Conv2DTranspose(64,3,strides=2,padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(32,3,strides=2,padding='same',activation='relu')(x)

    outputs = layers.Conv2D(1,3,padding='same',activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model
