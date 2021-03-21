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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



import sys
sys.path.append('../')
from UtilFunctions import *

fopOutput='../../../../dataPapers/analysisSEE/'
fopOutputAllSystems=fopOutput+'/RQ4_tuning_filter/'
fopResultTuning=fopOutputAllSystems+'/result_tuning/'
fopDataset='../../dataset_sorted/'
isUseBackup=True

createDirIfNotExist(fopOutputAllSystems)
createDirIfNotExist(fopResultTuning)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')






list_files = os.listdir(fopDataset)   # Convert to lower case
list_files =sorted(list_files)
random_seed=100

lstPrior=[]
fpPriorWork=fopOutput+'priorWork.txt'
fopPerProject=fopOutput+'/perProjects/'
fpRQ2PerProject= fopPerProject + 'rq2_pp3_textCount_train.xls'




fff=open(fpPriorWork,'r')
arrPriorResult=fff.read().strip().split('\n')
for item in arrPriorResult:
    lstPrior.append(float(item))

param_grid = {'C': [0.1,1, 10, 100], 'max_iter': [100,1000,5000,10000],'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}

lstAvgTuneMAE=[]

for cIndex in range(0,len(param_grid['C'])):
    for mIndex in range(0, len(param_grid['max_iter'])):
        for lIndex in range(0, len(param_grid['loss'])):
            lstMAE = []
            lstValMAE = []
            valCItem=param_grid['C'][cIndex]
            valMItem=param_grid['max_iter'][mIndex]
            valLItem = param_grid['loss'][lIndex]


            countBeaten=0
            for i in range(0,len(list_files)):
                fileName=list_files[i]
                systemName=fileName.replace('.csv','')
                fpSystemCsv=fopDataset+fileName
                dfSystem=pd.read_csv(fpSystemCsv)
                priorI=lstPrior[i]
                fpItemText = fopOutputAllSystems + systemName + '_text.txt'
                fpItemLabel = fopOutputAllSystems + systemName + '_label.txt'
                fpVectorItemReg=fopOutputAllSystems+systemName+'_vector.csv'

                lstTexts = []
                lstLabels = []
                if (isUseBackup and os.path.exists(fpItemText)):
                    fff = open(fpItemText, 'r')
                    lstTexts = fff.read().strip().split('\n')
                    fff.close()
                    fff = open(fpItemLabel, 'r')
                    arrLbls = fff.read().strip().split('\n')
                    for item in arrLbls:
                        lstLabels.append(int(item))
                    fff.close()
                else:
                    projectName=systemName.split('_')[3].lower()
                    dfWords = pd.read_excel(fpRQ2PerProject, sheet_name=projectName)
                    lstFilterWords = []
                    columnCount = dfWords['Count']
                    columnUniqueWords = dfWords['Text']
                    threshold = 30
                    for index in range(0, len(columnUniqueWords)):
                        itemCount = columnCount[index]
                        itemWord = str(columnUniqueWords[index])
                        '''
                        arrWs=itemWord.split('---')
                        if(len(arrWs)<2):
                            continue
                        '''
                        if itemCount <= 1:
                            lstFilterWords.append(itemWord)

                    setFilterWords = set(lstFilterWords)

                    lstTexts=[]
                    lstLabels=[]
                    columnTitle=dfSystem['title']
                    columnDescription=dfSystem['description']
                    columnSP=dfSystem['storypoint']

                    for j in range(0,len(columnTitle)):
                        strContent =' '.join([str(columnTitle[j]),str(columnDescription[j])])
                        # strContent = ' '.join([str(columnDescription[j])])
                        strContent=preprocessTextV3_FilerWordAndReplace(strContent,setFilterWords,ps,lemmatizer)
                        # intValue=int(columnSP[j])
                        # if intValue>=40:
                        #     continue

                        # print(strContent)
                        lstTexts.append(strContent)
                        lstLabels.append(columnSP[j])
                    fff = open(fpItemText, 'w')
                    fff.write('\n'.join(lstTexts))
                    fff.close()
                    fff = open(fpItemLabel, 'w')
                    fff.write('\n'.join(map(str, lstLabels)))
                    fff.close()

                if not os.path.exists(fpVectorItemReg):
                    vectorizer = TfidfVectorizer(ngram_range=(1,1))
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
                        for k in range(0, lenVectorOfWord):
                            strRow2 = ''.join([strRow2, ',', str(vector[k])])

                        strRow2 = ''.join([strRow2, '\n'])
                        #   csv.write(strRow)
                        csv.write(strRow2)
                    csv.close()

                dfVectors=pd.read_csv(fpVectorItemReg)
                all_label = dfVectors['story']
                all_data = dfVectors
                #all_data = dfVectors.drop(['no', 'story'], axis=1)



                X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2, shuffle=False)

               # # print('{}\t{}'.format(type(X_train),type(y_train)))
                X_train=X_train[X_train['story']<=20]
                y_train = X_train['story']
                X_train=X_train.drop(['no', 'story'], axis=1)
                X_test=X_test.drop(['no', 'story'], axis=1)


                regressor=LinearSVR(C=valCItem,loss=valLItem,max_iter=valMItem)
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
            fnTuneName='-'.join([str(valCItem),str(valMItem),str(valLItem)])
            newTuple=(fnTuneName,avgValue)
            lstAvgTuneMAE.append(newTuple)

            #lstMAE.append('Average\t{}'.format(avgValue))
            lstMAE.append('{}\n{}'.format(avgValue,countBeaten))
            fpRegressionResult=fopResultTuning+fnTuneName+'.txt'
            fff=open(fpRegressionResult,'w')
            fff.write('\n'.join(lstMAE))
            fff.close()

sortTuple(lstAvgTuneMAE,False)
fpTotalResult=fopOutputAllSystems+'result.txt'
lstTupleOut=[]
for item in lstAvgTuneMAE:
    strItem='{}\t{}'.format(item[0],item[1])
    lstTupleOut.append(strItem)
fff=open(fpTotalResult,'w')
fff.write('\n'.join(lstTupleOut))
fff.close()
















