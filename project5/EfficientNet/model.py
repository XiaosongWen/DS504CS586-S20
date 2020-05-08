import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, Input
from keras.layers import Lambda, Dense
import efficientnet.keras as efn


# Gem Pooling
gm_exp = tf.Variable(3.0, dtype=tf.float32)
def gem_pooling_2d(x):
    pool = (tf.reduce_mean(tf.abs(x**(gm_exp)), axis=[1, 2], keepdims=False) + 1.e-7) ** (1./gm_exp)
    return pool

def create_model(input_shape):
    input = Input(shape=input_shape)
    x_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input, pooling=None, classes=None)
    for layer in x_model.layers:
        layer.trainable = True
    lambda_layer = Lambda(gem_pooling_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x_model.output)

    grapheme_root = Dense(168, activation='softmax', name='root')(x)
    vowel_diacritic = Dense(11, activation='softmax', name='vowel')(x)
    consonant_diacritic = Dense(7, activation='softmax', name='consonant')(x)

    model = Model(inputs=x_model.input, outputs=[grapheme_root, vowel_diacritic, consonant_diacritic])

    return model