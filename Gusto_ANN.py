import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mth

gusto_train = pd.read_csv("/Users/shaileshdudala/Downloads/gusto_train.csv")
gusto_test = pd.read_csv("/Users/shaileshdudala/Downloads/gusto_test.csv")

YG_train = gusto_train.iloc[:,0]
YG_test = gusto_test.iloc[:,0]

gusto_train = gusto_train.iloc[:,1:]
gusto_test = gusto_test.iloc[:,1:]


#gusto_train.iloc[:,18]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [13])
X_train = onehotencoder.fit_transform(gusto_train).toarray()
X_train = X_train[:,1:]

onehotencoder = OneHotEncoder(categorical_features = [18])
X_train = onehotencoder.fit_transform(gusto_train).toarray()
X_train = X_train[:,1:]


onehotencoder = OneHotEncoder(categorical_features = [13])
X_test = onehotencoder.fit_transform(gusto_test).toarray()
X_test = X_test[:,1:]

onehotencoder = OneHotEncoder(categorical_features = [18])
X_test = onehotencoder.fit_transform(gusto_test).toarray()
X_test = X_test[:,1:]


#max(gusto_test.iloc[:,18])
#gusto_test.iloc[:,18].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))

# Adding the second hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, YG_train, batch_size = 10, epochs = 150, validation_data=(X_test,YG_test))


# Part 3 - Making predictions and evaluating the model

# =============================================================================
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(YG_test, y_pred)
# =============================================================================


# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
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
gs = clf.fit(X_train, YG_train, validation_data=(X_test,YG_test))
best_parameters = clf.best_params_
best_accuracy = clf.best_score_

metrics.roc_auc_score(YG_test, y_pred)
Test_acc= clf.score(X_test, YG_test)


# Predicting the Test set results
y_pred = clf.predict(X_test)
# =============================================================================
# y_pred = (y_pred > 0.5)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(YG_test, y_pred)
# 
# from statsmodels.stats.proportion import proportion_confint
# lower, upper = proportion_confint(2048, 2188, 0.05)
# print('lower=%.3f, upper=%.3f' % (lower, upper))
# =============================================================================


import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score


print("Original ROC area: {:0.3f}".format(roc_auc_score(YG_test, y_pred)))

n_bootstraps = 1000
rng_seed = 42  # control reproducibility
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
    if len(np.unique(YG_test[indices])) < 2:
        # We need at least one positive and one negative sample for ROC AUC
        # to be defined: reject the sample
        continue

    score = roc_auc_score(YG_test[indices], y_pred[indices])
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

np.mean(bootstrapped_scores)