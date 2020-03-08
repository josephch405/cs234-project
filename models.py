import numpy as np

class Model:
    # X: [n, feature_length]
    # Returns Y: [n, 3], corresponding to low, medium, high
    def predict(self, X):
        raise NotImplementedError

class FixedDose(Model):
    def predict(self, X):
        Y = np