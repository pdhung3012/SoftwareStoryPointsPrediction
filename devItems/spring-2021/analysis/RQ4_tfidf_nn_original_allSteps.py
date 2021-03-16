from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sys
sys.path.append('../')
from UtilFunctions import createDirIfNotExist

fopOutput='../../../../dataPapers/analysisSEE/'
fopOutputAllSystems=fopOutput+'/RQ4_TfidfML_nn/'
fopDataset='../../dataset_sorted/'

createDirIfNotExist(fopOutputAllSystems)

list_files = os.listdir(fopDataset)   # Convert to lower case
list_files =sorted(list_files)
random_seed=100

lstMAE=[]
lstValMAE=[]
lstPrior=[]
fpPriorWork=fopOutput+'priorWork.txt'
fff=open(fpPriorWork,'r')
arrPriorResult=fff.read().split('\n')
for item in arrPriorResult:
    lstPrior.append(float(item))

countBeaten=0
for i in range(0,len(list_files)):
    fileName=list_files[i]
    systemName=fileName.replace('.csv','')
    fpSystemCsv=fopDataset+fileName
    dfSystem=pd.read_csv(fpSystemCsv)
    priorI=lstPrior[i]
    fpVectorItemReg=fopOutputAllSystems+systemName+'_vector.csv'
    lstTexts=[]
    lstLabels=[]
    columnTitle=dfSystem['title']
    columnDescription=dfSystem['description']
    columnSP=dfSystem['storypoint']

    for j in range(0,len(columnTitle)):
        strContent =' '.join([str(columnTitle[j]),str(columnDescription[j])])
        lstTexts.append(strContent)
        lstLabels.append(columnSP[j])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(lstTexts)
    X = X.toarray()

    # X = PCA().fit(X)
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)


    lenVectorOfWord = len(X[0])
    columnTitleRow = "no,story,"
    for j in range(0, lenVectorOfWord):
        item = 'feature-' + str(j + 1)
        columnTitleRow = ''.join([columnTitleRow, item])
        if j != lenVectorOfWord - 1:
            columnTitleRow = ''.join([columnTitleRow, ","])
    columnTitleRow = ''.join([columnTitleRow, "\n"])
    csv = open(fpVectorItemReg, 'w')
    csv.write(columnTitleRow)
    corpusVector = []
    for j in range(0, len(lstTexts)):
        vector = X[j]
        corpusVector.append(vector)
        strReg = str(lstLabels[j])
        strRow2 = ''.join([str(j + 1), ',', '' + strReg, ])
        for j in range(0, lenVectorOfWord):
            strRow2 = ''.join([strRow2, ',', str(vector[j])])

        strRow2 = ''.join([strRow2, '\n'])
        #   csv.write(strRow)
        csv.write(strRow2)
    csv.close()

    dfVectors=pd.read_csv(fpVectorItemReg)
    all_label = dfVectors['story']
    all_data = dfVectors.drop(['no', 'story'], axis=1)



    X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2, shuffle=False)

    regressor=MLPRegressor(alpha=1,hidden_layer_sizes=(5,5))
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    maeAccuracy = mean_absolute_error(y_test, predicted)
    # strAcc='{}\t{}'.format(systemName,maeAccuracy)
    strAcc = '{}'.format(maeAccuracy)
    lstMAE.append(strAcc)
    lstValMAE.append(maeAccuracy)
    if maeAccuracy<priorI:
        countBeaten=countBeaten+1
    print('Finish {}'.format(systemName))

from statistics import mean
avgValue=mean(lstValMAE)
#lstMAE.append('Average\t{}'.format(avgValue))
lstMAE.append('{}\n{}'.format(avgValue,countBeaten))
fpRegressionResult=fopOutputAllSystems+'result_nn.txt'
fff=open(fpRegressionResult,'w')
fff.write('\n'.join(lstMAE))
fff.close()

















