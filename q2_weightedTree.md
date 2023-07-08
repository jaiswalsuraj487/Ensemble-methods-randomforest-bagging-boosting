The dataset for generation of weighted samples is as follows:

```python:
N = 200
P = 3
NUM_OP_CLASSES = 2
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=NUM_OP_CLASSES, class_sep=0.5)
X = pd.DataFrame(X)
y = pd.Series(y,dtype='category')
```
Weights are assigned to each target as follows:
```
weights=pd.Series(np.random.uniform(0,1,size=y.size))
```
Data has been shuffled using `sample(frac=1)` which shuffles data.

Default value of weight is 1 else the weight is given as per user.

The comparision results of sklearn and Implemented Weighted decision trees for different depths are as follows:

```
Depth: 1
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 2
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 3
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 4
Sklearn Accuracy: 0.8
Implemented Weighted DecisionTree Accuracy: 0.8
Depth: 5
Sklearn Accuracy: 0.7333333333333333
Implemented Weighted DecisionTree Accuracy: 0.8
```

So, it can be observed from above results that the accuracy of implemented weighted tree is in par with sklearn.