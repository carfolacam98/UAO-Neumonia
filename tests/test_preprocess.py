import pytest
import numpy as np
import cv2
from preprocess_img import CLAHEPreprocessStrategy


@pytest.fixture
def test_image():
    return np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)


@pytest.fixture
def strategy():
    return CLAHEPreprocessStrategy()


def test_preprocess_output_shape(strategy, test_image):
    processed_image = strategy.preprocess(test_image)
    assert processed_image.shape == (1, 512, 512, 1)


def test_preprocess_output_type(strategy, test_image):
    processed_image = strategy.preprocess(test_image)
    assert isinstance(processed_image, np.ndarray)


def test_preprocess_output_values(strategy, test_image):
    processed_image = strategy.preprocess(test_image)
    assert (processed_image >= 0).all() and (processed_image <= 1).all()
