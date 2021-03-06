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
    result = [0, 1, 0]
    if d < 21:
        result = [1, 0, 0]
    elif d > 49:
        result = [0, 0, 1]
    
    return np.array(result)

class Model:
    # X: [n, feature_length]
    # Returns Y: [n, 3], corresponding to low, medium, high
    def predict(self, X, X_float=None):
        raise NotImplementedError

    # x: [feature_length]
    # Returns Y: [3], corresponding to low, medium, high
    def predict_sample(self, x, x_float=None):
        raise NotImplementedError

    def feed_reward(self, r):
        pass

class FixedDose(Model):
    def predict(self, X):
        Y = np.zeros([X.shape[0], 3])
        Y[:, 1] = 1
        return Y

    def predict_sample(self, x, x_float):
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

class LinearSLModel(Model):
    def __init__(self, d):
        super(Model, LinearSLModel).__init__(self)
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.X = None
        self.Y = None
        self.x = None

    # x: [feature_length]
    # Returns Y: [3], corresponding to low, medium, high
    def predict_sample(self, x, x_float=None):

        self.x = x_float
        if self.X is not None:
            self.model.fit(self.X, self.Y)
            result = self.model.predict(x_float.reshape(1, -1))
            #print(result)
            b = [0, 0, 0]
            if result[0] < 0.5:
                b[0] = 1
            elif result[0] < 1.5:
                b[1] = 1
            else:
                b[2] = 1
            return np.array(b)
        else:
            return np.array([0, 1, 0])

    def feed_label(self, r):
        r = np.argmax(r)
        if self.X is None:
            self.X = self.x.reshape(1, -1)
            self.Y = np.array([r])
        else:
            self.X = np.append(self.X, self.x.reshape(1, -1), axis=0)
            #print(self.X.shape)
            self.Y = np.append(self.Y,r)

class SVMModel(Model):
    def __init__(self, d):
        super(Model, LinearSLModel).__init__(self)
        from sklearn import svm
        self.model = svm.SVC(gamma='scale')
        self.X = None
        self.Y = None
        self.x = None

    # x: [feature_length]
    # Returns Y: [3], corresponding to low, medium, high
    def predict_sample(self, x, x_float=None):

        self.x = x_float
        if self.X is not None:
            if not np.array_equal(self.Y, np.full(self.Y.shape, self.Y[0])): 
                self.model.fit(self.X, self.Y)
                result = self.model.predict(x_float.reshape(1, -1))
            else:
                result = self.Y[0]
            #print(result)
            b = [0, 0, 0]
            if result < 0.5:
                b[0] = 1
            elif result < 1.5:
                b[1] = 1
            else:
                b[2] = 1
            return np.array(b)
            
        else:
            return np.array([0, 1, 0])

    def feed_label(self, r):
        r = np.argmax(r)
        if self.X is None:
            self.X = self.x.reshape(1, -1)
            self.Y = np.array([r])
        else:
            self.X = np.append(self.X, self.x.reshape(1, -1), axis=0)
            #print(self.X.shape)
            self.Y = np.append(self.Y,r)


class OracleLinearModel(Model):
    def __init__(self, X, Y, X_float):
        super(Model, OracleLinearModel).__init__(self)
        from sklearn.linear_model import LogisticRegression
        # self.encoder = one_hot_encoder
        self.model = LogisticRegression(max_iter=1000)
        # self.n_numerical_feats = n_numerical_feats

        # X_numer = X[:, :n_numerical_feats].astype(np.float)
        # X_categ = self.encoder.transform(X[:, n_numerical_feats:]).toarray()
        # X_float = np.concatenate([X_numer, X_categ], axis=1)

        Y_labels = np.argmax(Y, axis=1)

        self.model.fit(X_float, Y_labels)
        # print(self.model.score(X_float, Y_labels))
    
    def predict(self, X, X_float):
        # n = self.n_numerical_feats
        # X_numer = X[:, :n].astype(np.float)
        # X_categ = self.encoder.transform(X[:, n:]).toarray()
        # X_float = np.concatenate([X_numer, X_categ], axis=1)

        p_labels = self.model.predict(X_float)
        results = np.zeros([X.shape[0], 3])
        results[np.arange(p_labels.size), p_labels] = 1
        return results

    
    def predict_sample(self, x, x_float):
        # n = self.n_numerical_feats

        # x_numer = x[:n].astype(np.float).reshape(1, -1)
        # x_categ = self.encoder.transform(x[n:].reshape(1, -1)).toarray()
        # x_float = np.concatenate([x_numer, x_categ], axis=1)

        result = self.model.predict(x_float)
        b = [0, 0, 0]
        b[result[0]] = 1
        return np.array(b)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LinearBasic(Model):
    def __init__(self, one_hot_encoder, n_numerical_feats):
        super(Model, LinearBasic).__init__(self)
        self.n_numerical_feats = n_numerical_feats
        # Low, medium, high
        self.beta = np.random.rand(3, n_numerical_feats + len(one_hot_encoder.get_feature_names()) + 1) - .5
        
    def predict_sample(self, x, x_float):
        # size [3] array of predictions
        bias_unit_added = np.append(x_float, 1)
        self.pred_r = self.beta.dot(bias_unit_added)
        self.x_float = bias_unit_added
        self.action = np.argmax(self.pred_r)

        self.n = [0, 0, 0]

        result = np.array([0, 0, 0])
        result[self.action] = 1
        return result

    def feed_reward(self, r):
        # sig = sigmoid(self.pred_r[self.action] + .5)
        # scale = sig * (1 - sig)
        self.n[self.action] += 1
        n = self.n[self.action]
        scale = 0.5/n
        # r_grad = r - self.pred_r[self.action]
        r_grad = 0
        # self.beta[self.action] += scale * self.x_float * 0.1
        if r > -1:
            # if self.pred_r[self.action] < 0:
            # print("positive")
            self.beta[self.action] += self.x_float * scale # * max(r_grad, 1)
        else:
            # print("negative")
            self.beta[self.action] -= self.x_float * scale # * min(r_grad, -1)
        # print(self.beta)


class LinUCB(Model):
    def __init__(self, d, alpha = 2):
        super(Model, LinUCB).__init__(self)
        self.d = d
        self.A = np.array([np.eye(d), np.eye(d), np.eye(d)])
        self.b = np.zeros([3, d])
        self.alpha = 2

    def predict_sample(self, x, x_float):
        # size [3] array of predictions
        self.x_float = x_float
        a_inv = np.linalg.inv(self.A)
        theta = np.einsum("bij,bj->bi", a_inv, self.b)
        p = theta.dot(x_float) + \
            self.alpha * np.sqrt(x_float.dot(a_inv).dot(x_float))
        self.action = np.argmax(p)

        result = np.array([0, 0, 0])
        result[self.action] = 1
        return result


    def feed_reward(self, r):
        self.A[self.action] += np.outer(self.x_float, self.x_float)
        self.b[self.action] += 2 * r * self.x_float