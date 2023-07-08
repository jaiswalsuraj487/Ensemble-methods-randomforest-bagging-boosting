## Q5. a
Random forest is implemented using sklearn Decision trees and is updated in `tree/randomForest.py`

The implementation is tested against `q5_RandomForest.py` and the results are as follows:
```
Criteria : information_gain
Accuracy:  93.33333333333333
Class:  0
Precision:  90.0
Recall:  100.0
Class:  3
Precision:  100.0
Recall:  85.71428571428571
Class:  2
Precision:  100.0
Recall:  85.71428571428571
Class:  4
Precision:  80.0
Recall:  100.0
Class:  1
Precision:  100.0
Recall:  100.0

Criteria : gini_index
Accuracy:  96.66666666666667
Class:  0
Precision:  100.0
Recall:  88.88888888888889
Class:  3
Precision:  100.0
Recall:  100.0
Class:  2
Precision:  87.5
Recall:  100.0
Class:  4
Precision:  100.0
Recall:  100.0
Class:  1
Precision:  100.0
Recall:  100.0
Criteria : gini_index
RMSE:  0.6644085768340983
MAE:  0.49385394601750776
```

The plots of trees are as follows:


Classification:

![Alt text](q5_RF_Classifier_fig1.png)

![Alt text](q5_RF_Classifier_fig2.png)

![Alt text](q5_RF_Classifier_fig3.png)

![Alt text](q5_RF_Classifier_fig4.png)

![Alt text](q5_RF_Classifier_fig5.png)

![Alt text](q5_RF_Classifier_fig6.png)

![Alt text](q5_RF_Classifier_fig7.png)

![Alt text](q5_RF_Classifier_fig8.png)

![Alt text](q5_RF_Classifier_fig9.png)

![Alt text](q5_RF_Classifier_fig10.png)


Regression:

![Alt text](q5_RF_Regressor_fig1.png)

![Alt text](q5_RF_Regressor_fig2.png)

![Alt text](q5_RF_Regressor_fig3.png)

![Alt text](q5_RF_Regressor_fig4.png)

![Alt text](q5_RF_Regressor_fig5.png)

![Alt text](q5_RF_Regressor_fig6.png)

![Alt text](q5_RF_Regressor_fig7.png)

![Alt text](q5_RF_Regressor_fig8.png)

![Alt text](q5_RF_Regressor_fig9.png)

![Alt text](q5_RF_Regressor_fig10.png)

## Q5. b
Classification data iris data is taken for plotting random forest. Two features are drawn for each decision surface and plots are as follows:

![Alt text](q5_partb_classifier_surface.png)

![Alt text](q5_partb_classifier.png)
