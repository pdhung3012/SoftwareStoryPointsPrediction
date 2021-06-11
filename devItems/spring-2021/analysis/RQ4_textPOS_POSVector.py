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
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from statistics import mean




import sys
from xgboost import *

sys.path.append('../')
from UtilFunctions import *

fopOutput='../../../../dataPapers/analysisSEE/'
fopOutputAllSystems=fopOutput+'/RQ4_POSVector/'
fopDataset='../../dataset_sorted/'
isUseBackup=True

createDirIfNotExist(fopOutputAllSystems)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#nltk.download('wordnet')






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


lstMAE = []
lstValMAE = []
lstRegressorName=[]

regressors = [DecisionTreeRegressor(),
                  AdaBoostRegressor(),  XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10),
                   LinearSVR(C=1.0,random_state=random_seed),
                   MLPRegressor(alpha=1,hidden_layer_sizes=(5,5)),
                   GradientBoostingRegressor(random_state=random_seed, max_depth=3)
                  ]

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
    lstIssueKeys=[]
    projectName = systemName.split('_')[3].lower()

    colIssueKey=dfSystem['issuekey']
    columnTitle = dfSystem['title']
    columnDescription = dfSystem['description']
    columnSP = dfSystem['storypoint']
    lstTextTrain, lstTextTest, lstLblTrain, lstLblTest = train_test_split(dfSystem, columnSP, test_size=0.2, shuffle=False)

    for idx2 in range(0,len(colIssueKey)):
        lstIssueKeys.append(str(colIssueKey[idx2]))



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

    # get dictionary of labels
    lstUniqueLabels=sorted(unique(lstLabels))

    dictLstStr={}
    for j in range(0,len(lstTexts)):
        scoreLbl=lstLabels[j]
        if not scoreLbl in dictLstStr.keys():
            lst=[lstTexts[j]]
            dictLstStr[scoreLbl]=lst
        else:
            dictLstStr[scoreLbl].append(lstTexts[j])
    lstKeys=list(dictLstStr.keys())
    lstKeyVectors=[]

    for key in lstKeys:
        lst=dictLstStr[key]
        lstKeyVectors.append(' '.join(lst))
    numKeys=len(dictLstStr.keys())

    for j in range(0,len(lstTexts)):
        lstKeyVectors.append(lstTexts[j])
    numTrainOnly=len(lstTextTrain)
    numTestOnly = len(lstTextTest)

    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    X = vectorizer.fit_transform(lstKeyVectors)

    dictSummaryDocumentVectors={}
    for j in range(0,numKeys):
        key=lstKeys[j]
        dictSummaryDocumentVectors[key]=X[j].todense()

    testIndex=numKeys+numTrainOnly
    minPredicted=[]
    y_test=[]

    matrixVector=[]
    for j in range(numKeys,len(lstKeyVectors)):
        vectorJ=X[j].todense()
        listScores=[]
        y_test.append(lstLabels[j-testIndex+numTrainOnly])
        lstItemVectors=[]
        for key in lstKeys:
            scoreItem= cosine_similarity(dictSummaryDocumentVectors[key],vectorJ)[0][0]
            lstItemVectors.append(scoreItem)
        matrixVector.append(lstItemVectors)

    lenVectorOfWord = numKeys
    columnTitleRow = "no,story,"
    for j in range(0, numKeys):
        item = 'feature-' + str(j + 1)
        columnTitleRow = ''.join([columnTitleRow, item])
        if j != lenVectorOfWord - 1:
            columnTitleRow = ''.join([columnTitleRow, ","])
    columnTitleRow = ''.join([columnTitleRow, "\n"])
    csv = open(fpVectorItemReg, 'w')
    csv.write(columnTitleRow)

    for j in range(0, len(matrixVector)):
        vector = matrixVector[j]
        strReg = str(lstLabels[j])
        strRow2 = ''.join([str(j + 1), ',', '' + strReg, ])
        for k in range(0, lenVectorOfWord):
            strRow2 = ''.join([strRow2, ',', str(vector[k])])
        strRow2 = ''.join([strRow2, '\n'])
        #   csv.write(strRow)
        csv.write(strRow2)
    csv.close()

    dfVectors = pd.read_csv(fpVectorItemReg)
    all_label = dfVectors['story']
    #all_data = dfVectors
    all_data = dfVectors.drop(['no', 'story'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2, shuffle=False)

    lstTupMAEForMLs = []
    for regressor in regressors:
        regressor.fit(X_train, y_train)
        predicted = regressor.predict(X_test)
        maeAccuracy = mean_absolute_error(y_test, predicted)
        newTupML = (regressor, maeAccuracy, predicted)
        lstTupMAEForMLs.append(newTupML)
    sortTuple(lstTupMAEForMLs, False)
    minPredicted = lstTupMAEForMLs[0][2]

    minMaeAccuracy=mean_absolute_error(y_test, minPredicted)
    lstValMAE.append(minMaeAccuracy)
    lstRegressorName.append(type(lstTupMAEForMLs[0][0]).__name__)
    # strAcc='{}\t{}'.format(systemName,maeAccuracy)
    if minMaeAccuracy<priorI:
        countBeaten=countBeaten+1
    print('Finish {}'.format(systemName))

avgValue=mean(lstValMAE)
fpRegressionResult=fopOutputAllSystems+'result.txt'
fff=open(fpRegressionResult,'w')
lstWriteToStr=[]
for i in range(0,len(lstRegressorName)):
    strItem='{}\t{}'.format(lstValMAE[i],lstRegressorName[i])
    lstWriteToStr.append(strItem)
lstWriteToStr.append('{}\n{}'.format(avgValue,countBeaten))
fff.write('\n'.join(lstWriteToStr))
fff.close()

















