import tensorflow as tf
import load_model
import grad_cam

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class PneumoniaIntegrator:
    def __init__(self, array):
        self.array = array
        self.model = load_model.read_model()
        self.grad_cam = grad_cam.ClassActivationHeatmap(self.array, self.model)

    def predict(self):
        return self.grad_cam.predict()
