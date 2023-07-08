Gradient Bossting is implmented using Decision Tree from sklearn and is tested using `q6_GradientBoosted.py`.

We are doing grid search for learning rate, number of estimators, and depth of decision tree to find optimal learning rate, number of estimators and depth of decision tree to minimize MSE and maximize R2 score.
```
C:\Users\HP\github-classroom\ES654\es654-spring2023-assignment2-pinki-kumari-sujeet-kumar-yadav>python -u "c:\Users\HP\github-classroom\ES654\es654-spring2023-assignment2-pinki-kumari-sujeet-kumar-yadav\q6_GradientBoosted.py"
----------------------
Learning Rate:  0.001 Estimators:  1 Depth:  1
MSE:  6807.583937870813
R2:  0.0010618585434940542
----------------------
Learning Rate:  0.001 Estimators:  1 Depth:  2
MSE:  6803.531336931662
R2:  0.001656533201517707
----------------------
Learning Rate:  0.001 Estimators:  1 Depth:  3
MSE:  6802.399560117484
R2:  0.0018226090130856187
----------------------
Learning Rate:  0.001 Estimators:  10 Depth:  1
MSE:  6743.104051575668
R2:  0.01052357335521703
----------------------
Learning Rate:  0.001 Estimators:  10 Depth:  2
MSE:  6702.940657379834
R2:  0.016417110139887114
----------------------
Learning Rate:  0.001 Estimators:  10 Depth:  3
MSE:  6691.724157405618
R2:  0.018063008264706304
----------------------
Learning Rate:  0.001 Estimators:  100 Depth:  1
MSE:  6153.615341313924
R2:  0.09702456431070072
----------------------
Learning Rate:  0.001 Estimators:  100 Depth:  2
MSE:  5791.971156103611
R2:  0.1500918747602774
----------------------
Learning Rate:  0.001 Estimators:  100 Depth:  3
MSE:  5687.752188019873
R2:  0.16538486317323353
----------------------
Learning Rate:  0.01 Estimators:  1 Depth:  1
MSE:  6742.782361107388
R2:  0.010570777896716566
----------------------
Learning Rate:  0.01 Estimators:  1 Depth:  2
MSE:  6702.438809987278
R2:  0.01649075073046824
----------------------
Learning Rate:  0.01 Estimators:  1 Depth:  3
MSE:  6691.171997279864
R2:  0.018144031696050056
----------------------
Learning Rate:  0.01 Estimators:  10 Depth:  1
MSE:  6150.964960156649
R2:  0.09741347862326188
----------------------
Learning Rate:  0.01 Estimators:  10 Depth:  2
MSE:  5787.988121139995
R2:  0.150676341375779
----------------------
Learning Rate:  0.01 Estimators:  10 Depth:  3
MSE:  5683.140483996032
R2:  0.16606158006553573
----------------------
Learning Rate:  0.01 Estimators:  100 Depth:  1
MSE:  2950.55421529334
R2:  0.5670385894626745
----------------------
Learning Rate:  0.01 Estimators:  100 Depth:  2
MSE:  1673.3504400658742
R2:  0.7544542096196843
----------------------
Learning Rate:  0.01 Estimators:  100 Depth:  3
MSE:  1292.2821033290184
R2:  0.8103718008740781
----------------------
Learning Rate:  0.1 Estimators:  1 Depth:  1
MSE:  6127.020771952688
R2:  0.10092702514453034
----------------------
Learning Rate:  0.1 Estimators:  1 Depth:  2
MSE:  5741.831087891324
R2:  0.15744937883361676
----------------------
Learning Rate:  0.1 Estimators:  1 Depth:  3
MSE:  5634.258504252694
R2:  0.1732344734798763
----------------------
Learning Rate:  0.1 Estimators:  10 Depth:  1
MSE:  2862.894234156068
R2:  0.5799017284906403
----------------------
Learning Rate:  0.1 Estimators:  10 Depth:  2
MSE:  1584.430919295237
R2:  0.7675021722490816
----------------------
Learning Rate:  0.1 Estimators:  10 Depth:  3
MSE:  1210.2066268257884
R2:  0.8224154752092838
----------------------
Learning Rate:  0.1 Estimators:  100 Depth:  1
MSE:  195.22129807786573
R2:  0.9713534195899135
----------------------
Learning Rate:  0.1 Estimators:  100 Depth:  2
MSE:  49.52603231749841
R2:  0.992732598947222
----------------------
Learning Rate:  0.1 Estimators:  100 Depth:  3
MSE:  13.853773099794774
R2:  0.9979671110222407
----------------------
Learning Rate:  1 Estimators:  1 Depth:  1
MSE:  3194.8227283588676
R2:  0.5311948691817403
----------------------
Learning Rate:  1 Estimators:  1 Depth:  2
MSE:  1167.5086017201268
R2:  0.8286809412295608
----------------------
Learning Rate:  1 Estimators:  1 Depth:  3
MSE:  601.3371088852197
R2:  0.9117603867361923
----------------------
Learning Rate:  1 Estimators:  10 Depth:  1
MSE:  483.7185533841821
R2:  0.9290196173685733
----------------------
Learning Rate:  1 Estimators:  10 Depth:  2
MSE:  138.4253993418564
R2:  0.9796875936586428
----------------------
Learning Rate:  1 Estimators:  10 Depth:  3
MSE:  43.52474451983985
R2:  0.9936132219896481
----------------------
Learning Rate:  1 Estimators:  100 Depth:  1
MSE:  51.82854482650237
R2:  0.9923947305364296
----------------------
Learning Rate:  1 Estimators:  100 Depth:  2
MSE:  0.036745157468493925
R2:  0.9999946080518957
----------------------
Learning Rate:  1 Estimators:  100 Depth:  3
MSE:  1.3724172884158133e-09
Optimal params on R2:  [1, 100, 3]
Min MSE:  1.3724172884158133e-09
Max R2:  0.9999999999997986
DTR MSE:  3194.8227283588676
DTR R2:  0.5311948691817403

```
