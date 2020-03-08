import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import models

print("Loading data...")

with open("data/warfarin.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    all_d = np.array(list(reader))[:-1, :-3]

# Removing all examples without target warfarin
all_d = all_d[all_d[:, 34] != "NA"]

print("Building X...")
numerical_feature_indices = [5, 6]
numerical_features = all_d[1:, numerical_feature_indices].copy()
numerical_features[(numerical_features == "NA") | (numerical_features == "")] = "0"

numerical_features = numerical_features.astype(np.float)

categorical_feature_indices = list(range(63))

# Mostly features that indicate results
bad_features_indices = [0, 33, 34, 35]

# Categorical features with a lot of text/descriptions
unhelpful_features_indices = [8, 12, 30]

for i in numerical_feature_indices + bad_features_indices + unhelpful_features_indices:
    categorical_feature_indices.remove(i)
categorical_features = all_d[1:, categorical_feature_indices].copy()

# encoder = OneHotEncoder()
# fitted_categorical_features = encoder.fit_transform(categorical_features).toarray()

X = np.append(numerical_features, categorical_features, axis=1)

print("Building Y...")
Y = all_d[1:, 34].copy().astype(np.float)

low = Y < 21
medium = (Y >= 21) & (Y <= 49)
high = Y > 49
Y_categ = np.stack([low, medium, high], axis=1).astype(int)

print("Running model...")

n_trials = 20 # normally 20 for real bandit runs
model = models.FixedDose()

def verifyPred(y):
    assert sum(y) == 1
    for j in y:
        assert (j == 0 or j == 1)

def calculateFractionCorrect(y, preds):
    return (y * preds).sum() / y.shape[0]

for i in range(n_trials):
    shuffled_order = np.arange(X.shape[0])
    np.random.shuffle(shuffled_order)
    X_shuffled, Y_categ_shuffled = X[shuffled_order], Y_categ[shuffled_order]
    preds = []
    for x in X_shuffled:
        pred = model.predict_sample(X)
        verifyPred(pred)
        preds.append(pred)
    preds = np.array(preds)

    # TODO: account for multiple trial scoring
    print(calculateFractionCorrect(Y_categ_shuffled, preds))