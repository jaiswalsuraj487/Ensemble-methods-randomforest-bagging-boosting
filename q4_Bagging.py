import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree

# Or use sklearn decision tree

########### BaggingClassifier ###################

# N = 30
# P = 2
# NUM_OP_CLASSES = 2
# n_estimators = 3
# X = pd.DataFrame(np.abs(np.random.randn(N, P)))
# y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

# criteria = "information_gain"
# tree = DecisionTree(criterion=criteria)
# Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators)
# Classifier_B.fit(X, y)
# y_hat = Classifier_B.predict(X)
# [fig1, fig2] = Classifier_B.plot()
# print("Criteria :", criteria)
# print("Accuracy: ", accuracy(y_hat, y))
# for cls in y.unique():
#     print("Precision: ", precision(y_hat, y, cls))
#     print("Recall: ", recall(y_hat, y, cls))



from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np
from ensemble.bagging import BaggingClassifier
from sklearn.model_selection import train_test_split
import time

# Generate a synthetic dataset for classification
X, y = make_classification(n_samples=4000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=80)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

#sequencial
classifier_s = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=10)

start_time = time.time()
classifier_s.fit(X, y, flag="sequencial")
end_time = time.time()

y_hat = classifier_s.predict(X)
print("Accuracy Squencial :", accuracy_score(y, y_hat))
print("Time Squencial :", round(end_time-start_time, 4), "seconds")

[fig1, fig2] = classifier_s.plot(X, y)
# fig1.savefig(fname='q4_bagging_sequential.png')
# fig2.savefig(fname='q4_bagging_sequential_combined.png')


#parallel
classifier_p = BaggingClassifier(base_model=DecisionTreeClassifier, num_estimators=10)

start_time = time.time()
classifier_p.fit(X, y, flag="parallel")
end_time = time.time()

y_hat = classifier_p.predict(X)
print("Accuracy Parallel :", accuracy_score(y, y_hat))
print("Time Parallel :", round(end_time-start_time, 4), "seconds")

[fig1, fig2] = classifier_p.plot(X, y)

# fig1.savefig(fname='q4_bagging_parallel.png')
# fig2.savefig(fname='q4_bagging_parallel_combined.png')





