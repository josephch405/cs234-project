import numpy as np

class Model:
    # X: [n, feature_length]
    # Returns Y: [n, 3], corresponding to low, medium, high
    def predict(self, X):
        raise NotImplementedError

    # x: [feature_length]
    # Returns Y: [3], corresponding to low, medium, high
    def predict_sample(self, x):
        raise NotImplementedError

class FixedDose(Model):
    def predict(self, X):
        Y = np.zeros([X.shape[0], 3])
        Y[:, 1] = 1
        return Y

    def predict_sample(self, x):
        return np.array([0, 1, 0])

class ClinicalAlgo(Model):
    def predict_sample(self, x):
        
        return 0