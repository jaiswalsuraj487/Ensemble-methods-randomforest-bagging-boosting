
# class BaggingClassifier():
#     def __init__(self, base_estimator, n_estimators=100):
#         '''
#         :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
#                                You can pass the object of the estimator class
#         :param n_estimators: The number of estimators/models in ensemble.
#         '''

#         pass

#     def fit(self, X, y):
#         """
#         Function to train and construct the BaggingClassifier
#         Inputs:
#         X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
#         y: pd.Series with rows corresponding to output variable (shape of Y is N)
#         """
#         pass

#     def predict(self, X):
#         """
#         Funtion to run the BaggingClassifier on a data point
#         Input:
#         X: pd.DataFrame with rows as samples and columns as features
#         Output:
#         y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
#         """
#         pass

#     def plot(self):
#         """
#         Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
#         Creates two figures
#         Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
#         The title of each of the estimator should be iteration number

#         Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

#         Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

#         This function should return [fig1, fig2]

#         """
#         pass
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import threading

class BaggingClassifier():
    def __init__(self, base_model, num_estimators):
        self.base_model = base_model
        self.num_estimators = num_estimators
        self.estimators = []
    
    def fit_single_model(self, X, y):
        estimator = self.base_model()
        sample_indices = np.random.choice(len(X), len(X), replace=True)
        sample_indices = sample_indices[:int(sample_indices.size/2)]
        estimator.fit(X[sample_indices], y[sample_indices])
        self.estimators.append(estimator)

    def fit(self, X, y, flag="sequencial"):
        if(flag == "parallel"):
            threads = []
            for i in range(self.num_estimators):
                threads.append(threading.Thread(target=self.fit_single_model, args=(X, y,)))
            for i in range(self.num_estimators):
                threads[i].start()
            for i in range(self.num_estimators):
                threads[i].join()

        elif(flag == "sequencial"):
            for i in range(self.num_estimators):
                self.fit_single_model(X, y)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for estimator in self.estimators:
            predictions += estimator.predict(X)
        return np.round(predictions / len(self.estimators))

    def plot(self, X, y):
        # Define the mesh grid for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot the decision surface of each estimator
        fig, axs = plt.subplots(1, self.num_estimators, figsize=(self.num_estimators * 3, 3), sharex=True, sharey=True)
        for i, clf in enumerate(self.estimators):
            # Create a subplot for the current estimator
            ax = axs[i]

            # Plot the decision surface
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

            # Set the title of the subplot
            ax.set_title(f'Estimator {i+1}')

        # Show the plot
        plt.show()
        plt.savefig(fname='q4_bagging_parallel.png')
        fig1 = plt
        # Plot the combined decision surface

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title('Combined decision surface')

        # Show the plot
        plt.show()
        
        plt.savefig(fname='q4_bagging_parallel_combined.png')
        fig2 = plt
        return [fig1, fig2]

