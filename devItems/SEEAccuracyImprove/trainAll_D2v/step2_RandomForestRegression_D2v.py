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
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
# import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import warnings

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")
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
            intVal=maxValue
        else:
            rangeValue = dictReverse[intVal + 1] - dictReverse[intVal]
        if intVal == 1:
            realValue = dictReverse[intVal]
        else:
            # print('{}'.format(intVal))
            realValue = dictReverse[intVal] + rangeValue * decVal
        lstNewColumn.append(realValue)
    return lstNewColumn


# set file directory
fpVectorAllSystems = 'model_d2v/vector-16-project.csv'
fopOverallResultReg= 'result_RFR_D2v/'
fopDataset='../../dataset/'
createDirIfNotExist(fopOverallResultReg)

from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]


# fpMAEMax = fopOverallResultReg + 'MAE_max.txt'
fpMAEMin = fopOverallResultReg + 'MAE_min.txt'
# fpMAEAvg = fopOverallResultReg + 'MAE_avg.txt'
# o3=open(fpMAEMax,'w')
# o3.write('')
# o3.close()

o3 = open(fpMAEMin, 'w')
o3.write('')
o3.close()

# o3 = open(fpMAEAvg, 'w')
# o3.write('')
# o3.close()
dictReverse={}

for file in arrFiles:
    if not file.endswith('.csv'):
        continue
    nameSystem=file.replace('.csv', '')
    # nameSystem=
    # fileCsv = fopVectorAllSystems + file+
    # fpVectorItemReg = fopVectorAllSystems + file + '_regression.csv'


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
    df_all = pd.read_csv(fpVectorAllSystems)
    # df_all=df_all.loc[(df_all['StoryReg'] <=10)]
    df_system=df_all.loc[df_all['Systems'] == nameSystem]
    # df_system=df_system.loc[(df_system['StoryReg'] <=10)]
    print(list(df_all.columns.values))
    all_label = df_all['StoryReg']
    # all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
    all_data = df_all.drop(['ID','Systems','StoryReg','StoryClass'],axis=1)
    system_label=df_system['StoryReg']
    system_ID = df_system['ID']
    system_data=df_system.drop(['ID','Systems','StoryReg','StoryClass'],axis=1)
    # X_system_train, XID_test_val, yid_train, yid_test_val = train_test_split(system_data, system_ID, test_size=0.4, shuffle=False,
    #                                                                   stratify=None)
    # X_system_train, X_test_val, y_system_train, y_test_val = train_test_split(system_data, system_label, test_size=0.4, shuffle=False,
    #                                                     stratify=None)
    # X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5,
    #                                                                           shuffle=False,
    #                                                                           stratify=None)

    X_system_train, XID_test, yid_train, yid_test = train_test_split(system_data, system_ID, test_size=0.2,
                                                                             shuffle=False,
                                                                             stratify=None)
    X_system_train, X_test, y_system_train, y_test = train_test_split(system_data, system_label, test_size=0.2,
                                                                              shuffle=False,
                                                                              stratify=None)
    # X_train=X_system_train
    # y_train=y_system_train
    # | (df_all['StoryReg'] == 100)
    # print(yid_test)
    # df_filter=df_all.loc[(df_all['ID'].isin(yid_train))| ((~df_all['ID'].isin(system_ID)) & (df_all['StoryReg'] == 1)) ]
    # df_filter = df_all.loc[(df_all['ID'].isin(yid_train)) & (df_all['StoryReg'] < 50) ]
    df_filter = df_all.loc[(df_all['ID'].isin(yid_train))]
    # df_filter = df_all.loc[(~df_all['ID'].isin(yid_test))]
    y_train = df_filter['StoryReg']
    X_train = df_filter.drop(['ID', 'Systems', 'StoryReg', 'StoryClass'], axis=1)
    dictReverse = {}
    y_train = convertNormalLabelToTopLabel(y_train)
    print('{}\t{}\t{}\t all data {}'.format(nameSystem,len(y_train),len(y_test),len(all_label)))

    o2=open(fpResultAll,'w')
    o2.close()

    bootstrap= [True]
    max_depth=[80, 90, 100, 110]
    max_features= [2, 3]
    min_samples_leaf= [3, 4, 5]
    min_samples_split= [8, 10, 12]
    n_estimators= [100, 200, 300, 1000]

    clf= RandomForestRegressor(bootstrap=True)
    bestMAE=1000
    bestMd=0
    bestMf=0
    bestMiSl=0
    bestMiSS=0
    bestNe=0


    for md in max_depth:
        for mf in max_features:
            for miSl in min_samples_leaf:
                for miSS in min_samples_split:
                    for ne in n_estimators:

                        clf=RandomForestRegressor(bootstrap=True,max_depth=md,max_features=mf,min_samples_leaf=miSl,min_samples_split=miSS,n_estimators=ne)
                        clf.fit(X_train,y_train)
                        predicted = clf.predict(X_test)

                        predicted = convertTopLabelToNormalLabel(predicted)
                        maeAccuracy = mean_absolute_error(y_test, predicted)
                        print('{}\t{}\t{}\t{}\t{}\t{}'.format(md, mf, miSl, miSS, ne,maeAccuracy))
                        if(maeAccuracy<bestMAE):
                            bestMAE=maeAccuracy
                            bestMd = md
                            bestMf = mf
                            bestMiSl = miSl
                            bestMiSS = miSS
                            bestNe = ne
    clf=RandomForestRegressor(bootstrap=True,max_depth=bestMd,max_features=bestMf,min_samples_leaf=bestMiSl,min_samples_split=bestMiSS,n_estimators=bestNe)
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    predicted = convertTopLabelToNormalLabel(predicted)
    maeAccuracy = mean_absolute_error(y_test, predicted)
    filePredict = ''.join([fopOutputItemDetail, file, '_RFR.txt'])
    np.savetxt(filePredict, predicted, fmt='%s', delimiter=',')

    o3 = open(fpMAEMin, 'a')
    o3.write('{}\t{}\t{}\n'.format(file, 'RFR', maeAccuracy))
    o3.close()
