import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

# from sklearn import datasets
from sklearn.datasets import make_classification
from random_forest_classification import RandomForestClassifierPlot

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Class: ', cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

Regressor_RF = RandomForestRegressor(10, criterion = criteria)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))




print('################## 5 (b) #####################')

N = 100
P = 10
NUM_OP_CLASSES = 2
n_estimators = 3
#X = pd.DataFrame(np.abs(np.random.randn(N, P)))
#y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, random_state=42, n_classes=NUM_OP_CLASSES)
# X = np.abs(np.random.randn(N, P))
# y = np.random.randint(NUM_OP_CLASSES, size=N)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a random forest classifier
clf = RandomForestClassifierPlot(n_estimators=n_estimators, max_depth=5, max_features=2)
clf.fit(X, y)

# Test the random forest classifier
y_pred = clf.predict(X)
# accuracy = accuracy_score(y_test, y_pred)

# # Print the accuracy of the random forest classifier
# print("Accuracy:", accuracy)
clf.plot()
