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
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.neural_network import *
from sklearn.discriminant_analysis import *


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



import sys
sys.path.append('../')
from UtilFunctions import createDirIfNotExist,preprocessTextV3


fopOutput='../../../../dataPapers/analysisSEE/'
fopOutputAllSystems=fopOutput+'/RQ4_pp4_binaryClass/'
fopDataset='../../dataset_sorted/'
fpResultDetails=fopOutputAllSystems+'result_details.txt'

createDirIfNotExist(fopOutputAllSystems)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')






list_files = os.listdir(fopDataset)   # Convert to lower case
list_files =sorted(list_files)
random_seed=100

lstMAE=[]
lstValMAE=[]
lstPrior=[]
fpPriorWork=fopOutput+'priorWork.txt'
fff=open(fpPriorWork,'r')
arrPriorResult=fff.read().strip().split('\n')
for item in arrPriorResult:
    lstPrior.append(float(item))
lstResultDetails=[]
countBeaten=0
o2 = open(fpResultDetails, 'w')
o2.write('')
o2.close()
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
    # minValue=columnSP.min()
    maxValue=columnSP.max()
    halfValue=maxValue //2
    print('old lbl {}/{} '.format(halfValue,maxValue))
    dfSystem.loc[dfSystem['storypoint'] < halfValue, 'storypoint'] = 0
    dfSystem.loc[dfSystem['storypoint'] >= halfValue, 'storypoint'] = 1

    columnSP = dfSystem['storypoint']
    #print(columnSP.idxmax(1))


    for j in range(0,len(columnTitle)):
        strContent =' '.join([str(columnTitle[j]),str(columnDescription[j])])
        # strContent = ' '.join([str(columnDescription[j])])
        strContent=preprocessTextV3(strContent,ps,lemmatizer)
        # intValue=int(columnSP[j])
        # if intValue>=40:
        #     continue

        # print(strContent)
        lstTexts.append(strContent)
        lstLabels.append(columnSP[j])

    #print(lstLabels)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(lstTexts)
    X = X.toarray()
    print('Numbers of n-grams: {}'.format(len(X[0])))
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
    all_data = dfVectors
        #.drop(['no', 'story'], axis=1)



    X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2, shuffle=False)

    # print('{}\t{}'.format(type(X_train),type(y_train)))
    # X_train=X_train[X_train['story']<=20]
    # y_train = X_train['story']
    # X_train=X_train.drop(['no', 'story'], axis=1)
    # X_test=X_test.drop(['no', 'story'], axis=1)


    classifier=LinearDiscriminantAnalysis()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    accuracyScore = accuracy_score(y_test, predicted)
    # strAcc='{}\t{}'.format(systemName,maeAccuracy)
    strAcc = '{}'.format(accuracyScore)
    lstMAE.append(strAcc)
    lstValMAE.append(accuracyScore)
    if accuracyScore<priorI:
        countBeaten=countBeaten+1
    o2 = open(fpResultDetails, 'a')
    o2.write('{}\n total accuracy ' + str(systemName) + '\n')

    o2.write('Confusion matrix:\n')
    o2.write(str(confusion_matrix(y_test, predicted)) + '\n')
    o2.write(str(classification_report(y_test, predicted)) + '\n')
    o2.close()
    print('Finish {}'.format(systemName))

from statistics import mean
avgValue=mean(lstValMAE)
#lstMAE.append('Average\t{}'.format(avgValue))
lstMAE.append('{}\n{}'.format(avgValue,countBeaten))
fpRegressionResult=fopOutputAllSystems+'result.txt'
fff=open(fpRegressionResult,'w')
fff.write('\n'.join(lstMAE))
fff.close()

















