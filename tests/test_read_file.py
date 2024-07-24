import pytest
from read_img import DICOMReadFileStrategy, JPGReadFileStrategy
from PIL import Image
import numpy as np
import cv2

@pytest.fixture
def dicom_strategy():
    return DICOMReadFileStrategy()

@pytest.fixture
def dicom_path():
    return "test_files/test.dcm"

@pytest.fixture
def jpg_strategy():
    return JPGReadFileStrategy()

@pytest.fixture
def jpg_path():
    return "test_files/test.jpeg"

def test_dicom_read_file(dicom_strategy, dicom_path):
    img_rgb, img_pil = dicom_strategy.read_file(dicom_path)
    assert isinstance(img_rgb, np.ndarray)
    assert isinstance(img_pil, Image.Image)
    assert img_rgb.shape == (1024, 1024, 3)
    assert img_pil.size == (1024, 1024)

def test_jpg_read_file(jpg_strategy, jpg_path):
    img_rgb, img_pil = jpg_strategy.read_file(jpg_path)
    assert isinstance(img_rgb, np.ndarray)
    assert isinstance(img_pil, Image.Image)
    assert img_rgb.shape == (230, 450, 3)
    assert img_pil.size == (450, 230)
