import numpy as np

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