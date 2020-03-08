import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import models

print("Loading data...")

with open("data/warfarin.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    all_d = np.array(list(reader))[:-1, :-3]

# Removing all examples without target warfarin
all_d = all_d[all_d[:, 34] != "NA"]

def feature_indices_to_names(indices):
    return list(map(lambda i: str(all_d[0][i]), indices))

print("Building X...")
numerical_feature_indices = [5, 6]
numerical_feature_names = feature_indices_to_names(numerical_feature_indices)
numerical_features = all_d[1:, numerical_feature_indices].copy()
missing_numerical_indices = (numerical_features == "NA") | (numerical_features == "")
# NB: see notebook, median/mean of height is about 168cm
numerical_features[missing_numerical_indices[:, 0]] = "168"
# NB: mean of weight is 75kg
numerical_features[missing_numerical_indices[:, 1]] = "75"

numerical_features = numerical_features.astype(np.float)

categorical_feature_indices = list(range(63))

# Mostly features that indicate results
bad_features_indices = [0, 33, 34, 35]

# Categorical features with a lot of text/descriptions
unhelpful_features_indices = [8, 12, 30]

for i in numerical_feature_indices + bad_features_indices + unhelpful_features_indices:
    categorical_feature_indices.remove(i)
categorical_feature_names = feature_indices_to_names(categorical_feature_indices)
categorical_features = all_d[1:, categorical_feature_indices].copy()

encoder = OneHotEncoder()
encoder.fit(categorical_features)

X = np.append(numerical_features, categorical_features, axis=1)
X_feature_names = numerical_feature_names + categorical_feature_names

print("Building Y...")
Y = all_d[1:, 34].copy().astype(np.float)

low = Y < 21
medium = (Y >= 21) & (Y <= 49)
high = Y > 49
Y_categ = np.stack([low, medium, high], axis=1).astype(int)

print("Running model...")

n_trials = 5 # normally 20 for real bandit runs
bestModel = models.OracleLinearModel(X, Y_categ, encoder, len(numerical_feature_indices))
model = models.LinearBasic(encoder, len(numerical_feature_indices))

def verifyPred(y):
    assert sum(y) == 1
    assert y.shape[0] == 3 and len(y.shape) == 1
    for j in y:
        assert (j == 0 or j == 1)

def calculateNumberCorrect(y, preds):
    return (y * preds).sum()

def calculateFractionCorrect(y, preds):
    return calculateNumberCorrect(y, preds) / y.shape[0]

def calculateReward(y, pred):
    if np.argmax(y) == np.argmax(pred):
        return 0
    return -1

for i in range(n_trials):
    shuffled_order = np.arange(X.shape[0])
    np.random.shuffle(shuffled_order)
    X_shuffled, Y_categ_shuffled = X[shuffled_order], Y_categ[shuffled_order]
    Y_best = bestModel.predict(X_shuffled)
    best_correctness = calculateFractionCorrect(Y_categ_shuffled, Y_best)

    preds = []
    for x, y in tqdm(list(zip(X_shuffled, Y_categ_shuffled))):
        pred = model.predict_sample(x)
        verifyPred(pred)
        preds.append(pred)
        r = calculateReward(y, pred)
        model.feed_reward(r)

    preds = np.array(preds)

    # TODO: account for multiple trial scoring
    print(calculateFractionCorrect(Y_categ_shuffled, preds))