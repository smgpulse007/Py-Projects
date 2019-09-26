#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:57:04 2019

@author: shaileshdudala
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mth


r_train = pd.read_csv("/Users/shaileshdudala/Downloads/readmission_train.csv")
r_test = pd.read_csv("/Users/shaileshdudala/Downloads/readmission_test.csv")

y_train = r_train.ix[:,'outcome']
y_test = r_test.ix[:,'outcome']

r_train = r_train.iloc[:,1:]
r_test = r_test.iloc[:,1:]

del r_train['outcome']
del r_test['outcome']

X_train = r_train.iloc[:,:].values
X_test = r_test.iloc[:, :].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_0 = LabelEncoder()
X_train[:, 0] = labelencoder_X_0.fit_transform(X_train[:, 0])
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])

labelencoder_X_2 = LabelEncoder()
X_test[:, 0] = labelencoder_X_2.fit_transform(X_test[:, 0])
labelencoder_X_3 = LabelEncoder()
X_test[:, 1] = labelencoder_X_3.fit_transform(X_test[:, 1])


onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:,1:]

# =============================================================================
# onehotencoder = OneHotEncoder(categorical_features = [18])
# X_train = onehotencoder.fit_transform(r_train).toarray()
# X_train = X_train[:,1:]
# =============================================================================


onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:,1:]

# =============================================================================
# onehotencoder = OneHotEncoder(categorical_features = [18])
# X_test = onehotencoder.fit_transform(r_test).toarray()
# X_test = X_test[:,1:]
# =============================================================================

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = "all", # resample all class
           random_state = 456,
           k_neighbors=1) 
           
X_train, y_train = sm.fit_sample(X_train, y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 21))
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [16, 32],
              'epochs': [100],
              'optimizer': ['adam', 'rmsprop']}
clf = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs=-1)
gs = clf.fit(X_train, y_train, validation_data=(X_test,y_test))
best_parameters = clf.best_params_
best_accuracy = clf.best_score_


metrics.roc_auc_score(y_test, y_pred)
Test_acc= clf.score(X_test, y_test)

# =============================================================================
# # Predicting the Test set results
# y_pred = clf.predict(X_test)
# y_pred = (y_pred > 0.5)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# 
# class_accuracy = 14316 / 14616 
# class_error = 300 / 14616 
# 
# #Wilson score interval
# CI_plus = class_error + 1.96 * mth.sqrt((class_error * (1 - class_error)) / 14616) #+ class_accuracy
# CI_minus =  class_error - 1.96 * mth.sqrt((class_error * (1 - class_error)) / 14616) 
# =============================================================================


# =============================================================================
# from statsmodels.stats.proportion import proportion_confint
# lower, upper = proportion_confint(14316, 14616, 0.05)
# print('lower=%.3f, upper=%.3f' % (lower, upper))
# =============================================================================


import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score


print("Original ROC area: {:0.3f}".format(roc_auc_score(y_test, y_pred)))

n_bootstraps = 1000
rng_seed = 42  # control reproducibility
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
    if len(np.unique(y_test[indices])) < 2:
        # We need at least one positive and one negative sample for ROC AUC
        # to be defined: reject the sample
        continue

    score = roc_auc_score(y_test[indices], y_pred[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
          
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# Computing the lower and upper bound of the 90% confidence interval
# You can change the bounds percentiles to 0.025 and 0.975 to get
# a 95% confidence interval instead.
confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    confidence_lower, confidence_upper))          