import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as st
from tqdm import tqdm

import models
import matplotlib.pyplot as plt

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

all_categorical_feature_indices = list(range(63))

# Mostly features that indicate results
bad_features_indices = [0, 33, 34, 35]

# Categorical features with a lot of text/descriptions
unhelpful_features_indices = [8, 12, 30]

for i in numerical_feature_indices + bad_features_indices + unhelpful_features_indices:
    all_categorical_feature_indices.remove(i)
all_categorical_features = all_d[1:, all_categorical_feature_indices].copy()

# categorical_feature_indices = all_categorical_feature_indices
categorical_feature_indices = [4, 2, 23, 24, 25, 26, 41, 42, 37]
categorical_feature_names = feature_indices_to_names(categorical_feature_indices)
categorical_features = all_d[1:, categorical_feature_indices].copy()

encoder = OneHotEncoder()
encoder.fit(categorical_features)

# NB: we keep all categorical features
X = np.append(numerical_features, all_categorical_features, axis=1)

X_subset = np.append(numerical_features, categorical_features, axis=1)
X_feature_names = numerical_feature_names + categorical_feature_names
n_num = len(numerical_feature_indices)
X_numer = X_subset[:, :n_num].astype(np.float)
def preprocess_num_feats(X_numer):
    for i in range(2):
        X_numer[:, i] -= np.min(X_numer[:, i])
        X_numer[:, i] /= np.max(X_numer[:, i])
preprocess_num_feats(X_numer)
X_categ = encoder.transform(X_subset[:, n_num:]).toarray()
X_float = np.concatenate([X_numer, X_categ], axis=1)

print("Building Y...")
Y = all_d[1:, 34].copy().astype(np.float)

low = Y < 21
medium = (Y >= 21) & (Y <= 49)
high = Y > 49
Y_categ = np.stack([low, medium, high], axis=1).astype(int)

print("Running model...")

n_trials = 20 # normally 20 for real bandit runs
bestModel = models.OracleLinearModel(X, Y_categ, X_float)
Y_best = bestModel.predict(X, X_float)
best_score = bestModel.model.score(X_float, np.argmax(Y_categ, axis=-1))
print("Best model:" + str(best_score))

clinModel = models.ClinicalAlgo()

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

def calculateConfidenceInterval(a):
    [lower, upper] = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
    return [lower, upper]

def computeCI(all):
    lows = []
    ups = []
    all_col = list(zip(*all))
    for sample in all_col:
        [low, up] = calculateConfidenceInterval(sample)
        lows.append(low)
        ups.append(up)
    return lows, ups

n_float_features = len(X_float[0])

all_regrets = []
all_frac_wrong = []
all_fixed_regrets = []
all_clin_regrets = []

for i in range(n_trials):
    # model = models.FixedDose()
    model = models.LinUCB(n_float_features, alpha=1)
    #model = models.LinearSLModel(n_float_features)

    shuffled_order = np.arange(X.shape[0])
    np.random.shuffle(shuffled_order)
    X_shuffled, X_float_shuffled = \
        X[shuffled_order], X_float[shuffled_order]
    Y_best_shuffled, Y_categ_shuffled = Y_best[shuffled_order], Y_categ[shuffled_order] 
    # best_correctness = calculateFractionCorrect(Y_categ_shuffled, Y_best_shuffled)

    preds = []
    regrets = []
    regret = 0
    n_wrong = 0
    n_total = 0
    cum_fraction_wrong = []

    fixed_regret = 0
    fixed_regrets = []
    clin_regret = 0
    clin_regrets = []

    for x, y, x_float, y_best in tqdm(list( \
        zip(X_shuffled, Y_categ_shuffled, X_float_shuffled, Y_best_shuffled))):
        pred = model.predict_sample(x, x_float)
        verifyPred(pred)
        preds.append(pred)
        r = calculateReward(y, pred)
        model.feed_reward(r)
        #model.feed_label(y) #only enable for the SL model, ow comment out

        n_total += 1
        if np.sum(pred.dot(y)) == 0:
            n_wrong += 1
        if np.sum(y_best.dot(y)) > 0:
            # possibly suffer regret
            if np.sum(pred.dot(y)) == 0:
                regret += 1
            # baselin models suffers regret
            clinPred = clinModel.predict_sample(x)
            if clinPred.dot(y) == 0:
                clin_regret += 1
            if not y[1]:
                fixed_regret += 1
        regrets.append(regret)
        fixed_regrets.append(fixed_regret)
        clin_regrets.append(clin_regret)
        cum_fraction_wrong.append(n_wrong / n_total)


    preds = np.array(preds)

    # account for multiple trial scoring
    print(calculateFractionCorrect(Y_categ_shuffled, preds))
    all_regrets += [regrets]
    all_frac_wrong += [cum_fraction_wrong]
    all_fixed_regrets += [fixed_regrets]
    all_clin_regrets += [clin_regrets]

# compute 95% confidence interval for each performance mentric
[regrets_low, regrets_up] = computeCI(all_regrets)
[fixed_regrets_low, fixed_regrets_up] = computeCI(all_fixed_regrets)
[clin_regrets_low, clin_regrets_up] = computeCI(all_clin_regrets)
[fracs_wrong_low, fracs_wrong_up] = computeCI(all_frac_wrong)

plt.plot(np.mean(all_regrets, axis=0))
plt.plot(np.mean(all_fixed_regrets, axis=0))
plt.plot(np.mean(all_clin_regrets, axis=0))
plt.legend(['all regrets', 'all fixed regrets', 'all clin regrets'])
plt.show()

plt.plot(np.mean(all_regrets, axis=0))
plt.plot(regrets_low, linestyle=':')
plt.plot(regrets_up, linestyle=':')
plt.plot(np.mean(all_fixed_regrets, axis=0))
plt.plot(fixed_regrets_low, linestyle=':')
plt.plot(fixed_regrets_up, linestyle=':')
plt.plot(np.mean(all_clin_regrets, axis=0))
plt.plot(clin_regrets_low, linestyle=':')
plt.plot(clin_regrets_up, linestyle=':')
plt.legend(['all regrets', 'all regret lower 95% CI bound', 'all regret upper 95% CI bound', \
    'all fixed regrets', 'fixed regret lower 95% CI bound', 'fixed regret upper 95% CI bound' \
    'all clinical regrets', 'clinical regret lower 95% CI bound', 'clinical regret upper 95% CI bound'])
plt.show()

plt.plot(np.mean(all_frac_wrong, axis=0))
plt.plot(fracs_wrong_low, linestyle=':')
plt.plot(fracs_wrong_up, linestyle=':')
plt.legend(['Fraction of incorrect dosage', 'lower 95% CI bound', 'upper 95% CI bound'])
plt.show()
