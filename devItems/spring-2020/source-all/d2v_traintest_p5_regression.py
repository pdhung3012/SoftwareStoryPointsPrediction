# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:24 2019

@author: hungphd
"""


# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import os
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

# set file directory
fopVectorAllSystems = 'data/testVectorSystems_D2v/'
fopOverallResultReg= 'result/d2v_regression_82/'
createDirIfNotExist(fopOverallResultReg)

from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopVectorAllSystems) if isfile(join(fopVectorAllSystems, f))]
fpMAEMax = fopOverallResultReg + 'MAE_max.txt'
fpMAEMin = fopOverallResultReg + 'MAE_min.txt'
fpMAEAvg = fopOverallResultReg + 'MAE_avg.txt'
o3=open(fpMAEMax,'w')
o3.write('')
o3.close()

o3 = open(fpMAEMin, 'w')
o3.write('')
o3.close()

o3 = open(fpMAEAvg, 'w')
o3.write('')
o3.close()

for file in arrFiles:
    if not file.endswith('_regression.csv'):
        continue
    file=file.replace('_regression.csv', '')
    # fileCsv = fopVectorAllSystems + file+
    fpVectorItemReg = fopVectorAllSystems + file + '_regression.csv'


    fopOutputItemDetail = fopOverallResultReg + "/details/"
    # fopOutputItemEachReg = fopOutputItemDetail + file + "/"
    fopOutputItemResult = fopOverallResultReg + "/result/"
    fopOutputItemChart = fopOverallResultReg + "/chart/"
    fpResultAll=fopOutputItemResult+file+'.txt'
    fpImage = fopOutputItemChart + file+'_MAE.png'



    createDirIfNotExist(fopOutputItemDetail)
    # createDirIfNotExist(fopOutputItemEachReg)
    createDirIfNotExist(fopOutputItemResult)
    createDirIfNotExist(fopOutputItemChart)
   # fnAll='_10cv.csv'
    # load data for 10-fold cv
    df_all = pd.read_csv(fpVectorItemReg)
    print(list(df_all.columns.values))
    all_label = df_all['story']
    # all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
    all_data = df_all.drop(['no','story'],axis=1)

    # create a list of classifiers
    random_seed = 2
    # classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
    #                RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
    #                LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=5)]
    classifiers = [DecisionTreeRegressor(),
                   RandomForestRegressor(random_state=2, n_estimators=1000),AdaBoostRegressor(), xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10),
                   LinearSVR(random_state=random_seed), MLPRegressor(alpha=1),
                   GradientBoostingRegressor(random_state=random_seed, max_depth=5)]

    # fit and evaluate for 10-cv
    index = 0
    # group = df_all['label']
    arrClassifierName = ['DTR', 'RFR', 'ABR', 'XGBR', 'LSVR', 'MLPR', 'GBR']

    arrXBar = []
    arrMAE = []
    arrStrMAEAvg = []
    arrIndex=[]
    o2=open(fpResultAll,'w')
    o2.close()
    k_fold = StratifiedKFold(10,shuffle=True)




    for classifier in classifiers:
        index=index+1
        try:
            filePredict = ''.join([fopOutputItemDetail, file,'_',arrClassifierName[index-1], '.txt'])
            print("********", "\n", "10 fold CV Results Regression with: ", str(classifier))

            X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2,shuffle = False, stratify = None)
            classifier.fit(X_train, y_train)

            predicted = classifier.predict(X_test)
            # cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
            # predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
            # weightAvg = precision_score(all_label, predicted, average='weighted') * 100
            # maeAccuracy = mean_absolute_error(all_label, predicted)
            # mqeAccuracy = mean_squared_error(all_label, predicted)
            maeAccuracy = mean_absolute_error(y_test, predicted)
            mqeAccuracy = mean_squared_error(y_test, predicted)
            # maeAccuracy = mean_absolute_error(all_label, predicted)

            print('{:.2f}'.format(maeAccuracy))

            np.savetxt(filePredict, predicted, fmt='%s', delimiter=',')
            o2 = open(fpResultAll, 'a')
            o2.write('Result for ' + str(classifier) + '\n')
            o2.write('MAE {}\nMQE {}\n'.format(maeAccuracy,mqeAccuracy))

            # o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
            # o2.write(str(confusion_matrix(all_label, predicted)) + '\n')
            # o2.write(str(classification_report(all_label, predicted)) + '\n')
            o2.close()

            strClassX = str(arrClassifierName[index - 1])
            arrIndex.append(index)
            arrXBar.append(strClassX)
            arrMAE.append(maeAccuracy)
            arrStrMAEAvg.append('{:.2f}'.format(maeAccuracy))
            # break
        except Exception as inst:
            print("Error ", index)
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)

    arrAlgm = np.array(arrMAE)
    bestMAE=np.amax(arrAlgm)
    worstMAE=np.amin(arrAlgm)
    avgMAE=np.average(arrAlgm)
    maxIndexMAE= np.argmax(arrAlgm)
    minIndexMAE = np.argmin(arrAlgm)

    print(maxIndexMAE)
    o3=open(fpMAEMax,'a')
    o3.write('{}\t{}\t{}\n'.format(file, arrClassifierName[maxIndexMAE], bestMAE))
    o3.close()

    o3 = open(fpMAEMin, 'a')
    o3.write('{}\t{}\t{}\n'.format(file, arrClassifierName[minIndexMAE], worstMAE))
    o3.close()

    o3 = open(fpMAEAvg, 'a')
    o3.write('{}\t{}\n'.format(file, avgMAE))
    o3.close()


    y_pos = np.arange(len(arrXBar))
    plt.bar(y_pos, arrMAE, align='center', alpha=0.5)
    plt.xticks(y_pos, arrIndex, rotation=90)
    plt.rcParams["figure.figsize"] = (40, 40)
    plt.ylabel('MAE Accuracy')
    plt.ylim(0, 10)
    for i in range(len(arrMAE)):
        plt.text(x=i - 0.5, y=arrMAE[i] + 1, s=arrStrMAEAvg[i])
        plt.text(x=i, y=arrMAE[i] - 1, s=arrXBar[i], rotation=90)

    plt.title(fpResultAll)
    plt.savefig(fpImage)
    plt.clf()