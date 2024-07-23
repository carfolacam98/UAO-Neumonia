import pytest
import tensorflow as tf
from load_model import read_model
import os


def test_read_model():
    assert os.path.exists('../conv_MLP_84.h5'), "El archivo de modelo no existe."
    assert isinstance(read_model(), tf.keras.Model)
