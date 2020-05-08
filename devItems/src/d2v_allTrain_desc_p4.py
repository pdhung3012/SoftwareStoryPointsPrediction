# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:24 2019

@author: hungphd
"""


# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict, StratifiedKFold
import os
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# set file directory
fop=''

fpInput=fop+'data/vectorD2vCategories_all_desc_exact.csv'
fopOutput=fop+"result/resultD2v_all_desc_82/"
fpImage = fopOutput + 'trainTest.png'

try:
    # Create target Directory
    os.mkdir(fopOutput)
    print("Directory ", fopOutput, " Created ")
except FileExistsError:
    print("Directory ", fopOutput, " already exists")

# fnAll='_10cv.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput)
print(list(df_all.columns.values))
all_label = df_all['story']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','story'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2, random_state = 10)


# o2 = open(fopOutput + 'result_all.txt', 'w')
#
# arrClassifierName = ['GNB', 'LR', 'DT', 'RF', 'AB', 'LDA',
#                      'QDA', 'SVC', 'MLP', 'GBo']
# arrXBar = []
# arrWeightAvg = []
# arrStrWeightAvg = []
# arrIndex = []
#
# random_seed = 1234
#
# classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
#                RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
#                LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=5)]
#
# index = 0
#
# for classifier in classifiers:
#     index = index + 1
#     filePredict = ''.join([fopOutput, 'predict_', str(index), '.txt'])
#     # o2=open(fpInput+'result_all.txt','w')
#     # print("********", "\n", "10 fold CV Results with: ", str(classifier))
#     classifier.fit(X_train, y_train)
#     predicted = classifier.predict(X_test)
#     # cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
#     # predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
#     np.savetxt(filePredict, predicted, fmt='%s', delimiter=',')
#     o2.write('Result for ' + str(classifier) + '\n')
#     # o2.write(str(sum(cross_val)/float(len(cross_val)))+'\n')
#     o2.write(str(confusion_matrix(y_test, predicted)) + '\n')
#     o2.write(str(classification_report(y_test, predicted)) + '\n')
#
#     weightAvg = precision_score(y_test, predicted, average='weighted') * 100
#     normalAvg = accuracy_score(y_test, predicted) * 100
#     print('{:.2f}'.format(normalAvg))
#
#     strClassX = str(arrClassifierName[index - 1])
#     arrIndex.append(index)
#     arrXBar.append(strClassX)
#     arrWeightAvg.append(normalAvg)
#     arrStrWeightAvg.append('{:.2f}'.format(normalAvg))
#
# o2.close()
#
# y_pos = np.arange(len(arrXBar))
# plt.bar(y_pos, arrWeightAvg, align='center', alpha=0.5)
# plt.xticks(y_pos, arrIndex, rotation=90)
# plt.rcParams["figure.figsize"] = (40, 40)
# plt.ylabel('Total Accuracy')
# plt.ylim(0, 100)
# for i in range(len(arrWeightAvg)):
#     plt.text(x=i - 0.5, y=arrWeightAvg[i] + 1, s=arrStrWeightAvg[i])
#     plt.text(x=i, y=arrWeightAvg[i] - 20, s=arrXBar[i], rotation=90)
#
# plt.title(fopOutput)
# plt.savefig(fpImage)
# plt.clf()

from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
# Instantiate an XGBRegressor
xgr = xgb.XGBRegressor(random_state=2)
# Fit the classifier to the training set
xgr.fit(X_train, y_train)
y_pred = xgr.predict(X_test)
mqeAccuracy=mean_squared_error(y_test, y_pred)
maeAccuracy=mean_absolute_error(y_test, y_pred)
# Accuracy
y_pred_rounded = [round(prediction,0) for prediction in y_pred ]
y_pred_rounded = [int(prediction) for prediction in y_pred_rounded]
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
acc_score = metrics.accuracy_score(y_test, y_pred_rounded)
print('Xgb Total accuracy classification score: {}'.format(acc_score))
print(str(confusion_matrix(y_test,y_pred_rounded)) + '\n')
print(str(classification_report( y_test,y_pred_rounded)) + '\n')
print('MAE: {} and MSE: {}'.format(maeAccuracy,mqeAccuracy))

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
mqeAccuracy=mean_squared_error(y_test, y_pred)
maeAccuracy=mean_absolute_error(y_test, y_pred)
# Accuracy
y_pred_rounded = [round(prediction,0) for prediction in y_pred ]
y_pred_rounded = [int(prediction) for prediction in y_pred_rounded]
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
acc_score = metrics.accuracy_score(y_test, y_pred_rounded)
print('RandomForest Total accuracy classification score: {}'.format(acc_score))
print(str(confusion_matrix(y_test,y_pred_rounded)) + '\n')
print(str(classification_report( y_test,y_pred_rounded)) + '\n')
print('MAE: {} and MSE: {}'.format(maeAccuracy,mqeAccuracy))


