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
fopVectorAllSystems = 'data/pretrainedVector/TFIDF4/'
fopOverallResultCate='result/'


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

createDirIfNotExist(fopOverallResultCate)

from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopVectorAllSystems) if isfile(join(fopVectorAllSystems, f))]

fpAccMax = fopOverallResultCate + 'Acc_max.txt'
fpAccMin = fopOverallResultCate + 'Acc_min.txt'
fpAccAvg = fopOverallResultCate + 'Acc_avg.txt'
o3=open(fpAccMax, 'w')
o3.write('')
o3.close()

o3 = open(fpAccMin, 'w')
o3.write('')
o3.close()

o3 = open(fpAccAvg, 'w')
o3.write('')
o3.close()

for file in arrFiles:
    if not file.endswith('_category.csv'):
        continue
    file=file.replace('_category.csv', '')
    # fileCsv = fopVectorAllSystems + file+
    fpVectorItemCate = fopVectorAllSystems + file + '_category.csv'

    # fopOutputItem = fopOverallResultCate + file+"/"
    fopOutputItemDetail = fopOverallResultCate + "/details/"
    fopOutputItemResult = fopOverallResultCate + "/result/"
    fopOutputItemChart = fopOverallResultCate  + "/chart/"
    fpResultAll=fopOutputItemResult+file+'.txt'
    fpImage = fopOutputItemChart + file+'.png'


    # createDirIfNotExist(fopOutputItem)
    createDirIfNotExist(fopOutputItemDetail)
    createDirIfNotExist(fopOutputItemResult)
    createDirIfNotExist(fopOutputItemChart)
    # fnAll='_10cv.csv'
    # load data for 10-fold cv
    df_all = pd.read_csv(fpVectorItemCate)
    print(list(df_all.columns.values))
    all_label = df_all['story']
    # all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
    all_data = df_all.drop(['no','story'],axis=1)

    # create a list of classifiers
    random_seed = 1234
    classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
                   RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
                   LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=5)]

    # fit and evaluate for 10-cv
    index = 0
    # group = df_all['label']
    arrClassifierName = ['GNB', 'LR', 'DT', 'RF', 'AB', 'LDA',
                         'QDA', 'SVC', 'MLP', 'GBo']

    arrXBar = []
    arrAccuracy = []
    arrStrWeightAvg = []
    arrIndex=[]
    o2=open(fpResultAll,'w')
    o2.close()
    k_fold = StratifiedKFold(10,shuffle=True)

    for classifier in classifiers:
        index=index+1
        try:
            filePredict = ''.join([fopOutputItemDetail, file,'_',arrClassifierName[index-1], '.txt'])
            print("********", "\n", "10 fold CV Results with: ", str(classifier))
            cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
            predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
            weightAvg = precision_score(all_label, predicted, average='weighted') * 100
            normalAvg = accuracy_score(all_label, predicted) * 100
            print('{:.2f}'.format(normalAvg))

            np.savetxt(filePredict, predicted, fmt='%s', delimiter=',')
            o2 = open(fpResultAll, 'a')
            o2.write('Result for ' + str(classifier) + '\n')
            o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
            o2.write(str(confusion_matrix(all_label, predicted)) + '\n')
            o2.write(str(classification_report(all_label, predicted)) + '\n')
            o2.close()

            strClassX = str(arrClassifierName[index - 1])
            arrIndex.append(index)
            arrXBar.append(strClassX)
            arrAccuracy.append(normalAvg)
            arrStrWeightAvg.append('{:.2f}'.format(normalAvg))
        except Exception as inst:
            print("Error ", index)
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)

    arrAlgm = np.array(arrAccuracy)
    bestAcc = np.amax(arrAlgm)
    worstAcc = np.amin(arrAlgm)
    avgMAE = np.average(arrAlgm)
    maxIndexMAE = np.argmax(arrAlgm)
    minIndexMAE = np.argmin(arrAlgm)

    print(maxIndexMAE)
    o3 = open(fpAccMax, 'a')
    o3.write('{}\t{}\t{}\n'.format(file, arrClassifierName[maxIndexMAE], bestAcc))
    o3.close()

    o3 = open(fpAccMin, 'a')
    o3.write('{}\t{}\t{}\n'.format(file, arrClassifierName[minIndexMAE], worstAcc))
    o3.close()

    o3 = open(fpAccAvg, 'a')
    o3.write('{}\t{}\n'.format(file, avgMAE))
    o3.close()


    y_pos = np.arange(len(arrXBar))
    plt.bar(y_pos, arrAccuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, arrIndex, rotation=90)
    plt.rcParams["figure.figsize"] = (40, 40)
    plt.ylabel('Total Accuracy')
    plt.ylim(0, 100)
    for i in range(len(arrAccuracy)):
        plt.text(x=i - 0.5, y=arrAccuracy[i] + 1, s=arrStrWeightAvg[i])
        plt.text(x=i, y=arrAccuracy[i] - 20, s=arrXBar[i], rotation=90)

    plt.title(fpResultAll)
    plt.savefig(fpImage)
    plt.clf()