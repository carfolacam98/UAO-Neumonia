import tensorflow as tf
import os

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def read_model():
    model_path = os.path.join(os.path.dirname(__file__), 'conv_MLP_84.h5')
    model = tf.keras.models.load_model(model_path)
    return model
