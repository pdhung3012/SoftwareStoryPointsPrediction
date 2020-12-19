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
from sklearn.model_selection import GridSearchCV


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")


# Ppython program to check if two
# to get unique values from list
# using numpy.unique
import numpy as np


# function to get unique values
def unique(list1):
    x = np.array(list1)
    return np.unique(x)
def convertNormalLabelToTopLabel(originColumn):
    lstUnique=unique(originColumn)
    lstUnqSort=sorted(lstUnique)
    dictTop={}
    for i in range(1,len(lstUnqSort)+1):
        valInt=int(lstUnqSort[i-1])
        dictTop[valInt]=i
        dictReverse[i]=valInt
    lstNewColumn=[]
    for item in originColumn:
        newScore=dictTop[item]
        lstNewColumn.append(newScore)
    # print(dictReverse)
    return lstNewColumn

import math
def convertTopLabelToNormalLabel(topColumn):

    minValue=min(dictReverse.keys())
    maxValue=max(dictReverse.keys())
    lstNewColumn=[]
    for item in topColumn:
        rangeValue=0
        decVal, intVal = math.modf(item)
        intVal=int(intVal)
        if intVal <= minValue:
            intVal = 1
            rangeValue = dictReverse[minValue]
        elif intVal >= maxValue:
            rangeValue = 0
        else:
            rangeValue = dictReverse[intVal + 1] - dictReverse[intVal]
        if intVal == 1:
            realValue = dictReverse[intVal]
        else:
            realValue = dictReverse[intVal] + rangeValue * decVal
        lstNewColumn.append(realValue)
    return lstNewColumn



# set file directory
nameSystem='mesos'
fopVectorAllSystems = 'vector_tfidf_dependency_'+nameSystem+'/'
fopOverallResultReg= 'result_tfidf_reg_depend_'+nameSystem+'/'
createDirIfNotExist(fopOverallResultReg)

dictReverse={}

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
    random_seed = 100

    param_DTR = {'max_depth': range(2, 16, 2),
                  'min_samples_split': range(2, 16, 2)}
    param_ADA = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
        'loss': ['linear', 'square', 'exponential']
    }

    param_XGBR={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 # "min_child_weight" : [ 1, 3, 5, 7 ],
 # "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

    param_LinearSVR = { 'C': [1,1.5, 10]}

    param_MLPR = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    param_GBR = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
               'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750]}

    # classifiers = [DecisionTreeRegressor(),
    #               AdaBoostRegressor(),
    #                xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
    #             max_depth = 5, alpha = 10, n_estimators = 10),
    #                LinearSVR(C=1.0,random_state=random_seed),
    #                MLPRegressor(alpha=1,hidden_layer_sizes=(5,5)),
    #                GradientBoostingRegressor(random_state=random_seed, max_depth=3)
    #               ]

    scoringFunction='neg_mean_absolute_error'
    classifiers = [
        GridSearchCV(
            estimator=DecisionTreeRegressor(),
            param_grid=param_DTR, scoring= scoringFunction, cv=2),
        # DecisionTreeRegressor(),
        GridSearchCV(
            estimator=AdaBoostRegressor(),
            param_grid=param_ADA, scoring= scoringFunction, cv=2),
                  # AdaBoostRegressor(),
        GridSearchCV(
            estimator=xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10),
            param_grid=param_XGBR, scoring= scoringFunction, cv=2),
                #    xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                # max_depth = 5, alpha = 10, n_estimators = 10),
        GridSearchCV(
        estimator = LinearSVR(C=1.0,random_state=random_seed),
                    param_grid = param_LinearSVR, scoring =  scoringFunction, cv = 2),
                   # LinearSVR(C=1.0,random_state=random_seed),
        GridSearchCV(
        estimator = MLPRegressor(alpha=1,hidden_layer_sizes=(5,5)),
                    param_grid = param_MLPR, scoring =  scoringFunction, cv = 2),
                   # MLPRegressor(alpha=1,hidden_layer_sizes=(5,5)),
        GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=random_seed, max_depth=3),
            param_grid=param_GBR, scoring= scoringFunction, cv=2),
                   # GradientBoostingRegressor(random_state=random_seed, max_depth=3)
                  ]

    # fit and evaluate for 10-cv
    index = 0
    # group = df_all['label']
    # 'RFR',
    arrClassifierName = ['DTR',  'ABR', 'XGBR', 'LSVR', 'MLPR', 'GBR']

    arrXBar = []
    arrMAE = []
    arrStrMAEAvg = []
    arrIndex=[]
    o2=open(fpResultAll,'w')
    o2.close()
    k_fold = StratifiedKFold(10,shuffle=True)




    for classifier in classifiers:
        index=index+1
        # try:
        filePredict = ''.join([fopOutputItemDetail, file,'_',arrClassifierName[index-1], '.txt'])
        print("********", "\n", "10 fold CV Results Regression with: ", str(classifier))

        X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2,shuffle = False, stratify = None)
        # X_train, X_val, y_train, y_val = train_test_split(all_data, all_label, test_size=0.25, shuffle=False,
        #                                                     stratify=None)
        dictReverse={}
        y_train=convertNormalLabelToTopLabel(y_train)
        # print(dictReverse)


        classifier.fit(X_train, y_train)

        predicted = classifier.best_estimator_.predict(X_test)


        predicted=convertTopLabelToNormalLabel(predicted)

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
        # except Exception as inst:
        #     print("Error ", index)
        #     print(type(inst))  # the exception instance
        #     print(inst.args)  # arguments stored in .args
        #     print(inst)

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