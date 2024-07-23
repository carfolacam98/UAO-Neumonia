import cv2
import numpy as np

from preprocess_img import CLAHEPreprocessStrategy


class BacteriaLabels:
    bacteria_label_mapping = {
        0: 'bacteriana',
        1: 'normal',
        2: 'viral',
    }


class BacteriaPredictionLabels:
    def __init__(self, entry):
        self.entry = entry

    def get_label(self):
        if self.entry in BacteriaLabels.bacteria_label_mapping:
            return BacteriaLabels.bacteria_label_mapping[self.entry]


class ClassActivationHeatmap(CLAHEPreprocessStrategy):
    def __init__(self, array, model):
        self.model = model
        self.array = array
        self.img = CLAHEPreprocessStrategy().preprocess(self.array)

    def create_heatmap(self, conv_layer_output_value):
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (self.img.shape[1], self.img.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img2 = cv2.resize(self.array, (512, 512))
        hif = 0.8
        transparency = heatmap * hif
        transparency = transparency.astype(np.uint8)
        superimposed_img = cv2.add(transparency, img2)
        superimposed_img = superimposed_img.astype(np.uint8)
        return superimposed_img[:, :, ::-1]

    def predict(self):
        batch_array_img = CLAHEPreprocessStrategy().preprocess(self.array)
        model = self.model
        prediction = np.argmax(model.predict(batch_array_img))
        proba = np.max(model.predict(batch_array_img)) * 100
        label = BacteriaPredictionLabels(prediction).get_label()
        heatmap = self.create_heatmap(self.array)
        return label, proba, heatmap
