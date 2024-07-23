from abc import ABC, abstractmethod
from PIL import Image

import cv2
import pydicom
import numpy as np


class ReadFileStrategy(ABC):
    @abstractmethod
    def read_file(self, path):
        pass


class DICOMReadFileStrategy(ReadFileStrategy):
    def read_file(self, path):
        img = pydicom.read_file(path)
        img_array = img.pixel_array
        img2show = Image.fromarray(img_array)
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        return img_RGB, img2show


class JPGReadFileStrategy(ReadFileStrategy):
    def read_file(self, path):
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img2show = Image.fromarray(img_array)
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        return img2, img2show
