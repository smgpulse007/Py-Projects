import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

# =============================================================================
# onehotencoder = OneHotEncoder(categorical_features = [18])
# X_train = onehotencoder.fit_transform(gusto_train).toarray()
# X_train = X_train[:,1:]
# =============================================================================


onehotencoder = OneHotEncoder(categorical_features = [13])
X_test = onehotencoder.fit_transform(gusto_test).toarray()
X_test = X_test[:,1:]

# =============================================================================
# onehotencoder = OneHotEncoder(categorical_features = [18])
# X_test = onehotencoder.fit_transform(gusto_test).toarray()
# X_test = X_test[:,1:]
# =============================================================================

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

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(YG_test, y_pred)

print(cm)
