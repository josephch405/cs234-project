import numpy as np
import math

def getAgeInDecades(x):
    decadeStr = x[5]
    if decadeStr == "NA":
        # TODO: impute properly? assuming average person is about 40
        return 4
    return int(decadeStr[0])

def getHeightInCm(x):
    return float(x[0])

def getWeightInKg(x):
    return float(x[1])

def isAsian(x):
    return x[3] == "Asian"

def isBlack(x):
    return x[3] == "Black or African American"

def missingOrMixedRace(x):
    return x[3] == "Unknown"

def enzymeInducerStatus(x):
    carbamazepine = x[21] == 1
    phenytoin = x[22] == 1
    rifampin_or_picin = x[23] == 1
    return carbamazepine or phenytoin or rifampin_or_picin

def amiodaroneStatus(x):
    return x[20] == 1

def dosageToAction(d):
    if d < 21:
        return [1, 0, 0]
    elif d > 49:
        return [0, 0, 1]
    return [0, 1, 0]

class Model:
    # X: [n, feature_length]
    # Returns Y: [n, 3], corresponding to low, medium, high
    def predict(self, X):
        raise NotImplementedError

    # x: [feature_length]
    # Returns Y: [3], corresponding to low, medium, high
    def predict_sample(self, x):
        raise NotImplementedError

    def feed_reward(self, r):
        pass

class FixedDose(Model):
    def predict(self, X):
        Y = np.zeros([X.shape[0], 3])
        Y[:, 1] = 1
        return Y

    def predict_sample(self, x):
        return np.array([0, 1, 0])

class ClinicalAlgo(Model):
    def predict_sample(self, x):
        dosage = 4.0376
        # Age in decades
        dosage -= 0.2546 * getAgeInDecades(x)
        # Height in cm
        dosage += 0.0118 * getHeightInCm(x)
        # Weight in kg
        dosage += 0.0134 * getWeightInKg(x)
        # Asian
        dosage -= 0.6752 * isAsian(x)
        # Black or African American
        dosage += 0.4060 * isBlack(x)
        # Missing or Mixed
        dosage += 0.0443 * missingOrMixedRace(x)
        # Enzyme Inducer Status
        dosage += 1.2799 * enzymeInducerStatus(x)
        # Amiodarone Status
        dosage -= -0.5695 * amiodaroneStatus(x)

        dosage = dosage * dosage

        return dosageToAction(dosage)

class OracleLinearModel(Model):
    def __init__(self, X, Y, one_hot_encoder, n_numerical_feats):
        super(Model, OracleLinearModel).__init__(self)
        from sklearn.linear_model import LogisticRegression
        self.encoder = one_hot_encoder
        self.model = LogisticRegression()
        self.n_numerical_feats = n_numerical_feats

        X_numer = X[:, :n_numerical_feats].astype(np.float)
        X_categ = self.encoder.transform(X[:, n_numerical_feats:]).toarray()
        X_float = np.concatenate([X_numer, X_categ], axis=1)

        Y_labels = np.argmax(Y, axis=1)

        self.model.fit(X_float, Y_labels)
        # print(self.model.score(X_float, Y_labels))
    
    def predict(self, X):
        n = self.n_numerical_feats
        X_numer = X[:, :n].astype(np.float)
        X_categ = self.encoder.transform(X[:, n:]).toarray()
        X_float = np.concatenate([X_numer, X_categ], axis=1)

        p_labels = self.model.predict(X_float)
        results = np.zeros([X.shape[0], 3])
        results[np.arange(p_labels.size), p_labels] = 1
        return results

    
    def predict_sample(self, x):
        n = self.n_numerical_feats

        x_numer = x[:n].astype(np.float).reshape(1, -1)
        x_categ = self.encoder.transform(x[n:].reshape(1, -1)).toarray()
        x_float = np.concatenate([x_numer, x_categ], axis=1)

        result = self.model.predict(x_float)
        b = [0, 0, 0]
        b[result[0]] = 1
        return np.array(b)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LinearBasic(Model):
    def __init__(self, one_hot_encoder, n_numerical_feats):
        super(Model, OracleLinearModel).__init__(self)
        self.n_numerical_feats = n_numerical_feats
        # Low, medium, high
        self.encoder = one_hot_encoder
        self.beta = np.random.rand(3, n_numerical_feats + len(self.encoder.get_feature_names()))
        
    def predict_sample(self, x):
        n = self.n_numerical_feats
        # size [3] array of predictions
        x_numer = x[:n].astype(np.float).reshape(1, -1)
        x_categ = self.encoder.transform(x[n:].reshape(1, -1)).toarray()
        x_float = np.concatenate([x_numer, x_categ], axis=1).reshape(-1)

        self.pred_r = self.beta.dot(x_float)
        self.x_float = x_float
        self.action = np.argmax(self.pred_r)
        result = np.array([0, 0, 0])
        result[self.action] = 1
        return result

    def feed_reward(self, r):
        # sig = sigmoid(self.pred_r[self.action] + .5)
        # scale = sig * (1 - sig)
        scale = 0.01
        # self.beta[self.action] += scale * self.x_float * 0.1
        if r > -1:
            # if self.pred_r[self.action] < 0:
            self.beta[self.action] += self.x_float * scale
        else:
            self.beta[self.action] -= self.x_float * scale