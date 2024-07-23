import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def read_model():
    model = tf.keras.models.load_model('conv_MLP_84.h5')
    return model
