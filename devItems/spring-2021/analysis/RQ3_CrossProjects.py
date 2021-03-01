from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from numpy import unique

# Bring your packages onto the path
import sys, os
import operator
sys.path.append(os.path.abspath(os.path.join('..')))
import pandas as pd
import io
import nltk
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from UtilFunctions import createDirIfNotExist,preprocess,intersection,diff
import operator

fopDataset='../../dataset_sorted/'
fopOutputLabelAna='../../../../dataPapers/analysisSEE/'
fopPerProject=fopOutputLabelAna+'perProjects/rq3_relations/'


def extractOverlapAndMatchBetweenSystems(systemName1,systemName2,dictAll,numMax):
    lstExcelSheetDataInter=['No,Label,InterWord,IndexSys1,IndexSys2,NumAppSys1,NumAppSys2']
    lstExcelSheetDataSubtract = ['No,Label,DiffWord,IndexSys1,IndexSys2,NumAppSys1,NumAppSys2']

    dictLblSys1=dictAll[systemName1];
    dictLblSys2 = dictAll[systemName2];
    for intValue in sorted(dictLblSys1.keys()):
        if(intValue in dictLblSys2.keys()):
            dictWord1={}
            listAllWord1=dictLblSys1[intValue]
            for i in range(0,min(len(listAllWord1),numMax)):
                keyIt=list(listAllWord1.keys())[i]
                #print('key {}'.format(keyIt))
                dictWord1[keyIt]=listAllWord1[keyIt]

            dictWord2 = {}
            listAllWord2 = dictLblSys2[intValue]
            for i in range(0, min(len(listAllWord2), numMax)):
                keyIt = list(listAllWord2.keys())[i]
                dictWord2[keyIt] = listAllWord2[keyIt]

            lstKeyWord1=list(dictWord1.keys())
            lstKeyWord2=list(dictWord2.keys())
           # print('{}\nb{}'.format(lstKeyWord1,lstKeyWord2))
            lstInter=intersection(lstKeyWord1,lstKeyWord2)
            listDistinct1=diff(lstKeyWord1,lstInter)
            listDistinct2 = diff(lstKeyWord2, lstInter)
            for i in range(0,len(lstInter)):
                strInter='{},{},{},{},{},{},{}'.format((i+1),intValue,lstInter[i],lstKeyWord1.index(lstInter[i]),lstKeyWord2.index(lstInter[i]),dictWord1[lstInter[i]],dictWord2[lstInter[i]])
                #print(strInter)
                lstExcelSheetDataInter.append(strInter)

            index=0
            for i in range(0,len(listDistinct1)):
                index=index+1
                strD1='{},{},{},{},{},{},{}'.format(index,intValue,listDistinct1[i],lstKeyWord1.index(listDistinct1[i]),-1,dictWord1[listDistinct1[i]],-1)
                lstExcelSheetDataSubtract.append(strD1)
            for i in range(0,len(listDistinct2)):
                index=index+1
                strD2='{},{},{},{},{},{},{}'.format(index,intValue,listDistinct2[i],-1,lstKeyWord2.index(listDistinct2[i]),-1,dictWord2[listDistinct2[i]])
                lstExcelSheetDataSubtract.append(strD2)

    dfItemInter = pd.read_csv(io.StringIO('\n'.join(lstExcelSheetDataInter)), sep=",")
    dfItemSubtract = pd.read_csv(io.StringIO('\n'.join(lstExcelSheetDataSubtract)), sep=",")
    return dfItemInter,dfItemSubtract


createDirIfNotExist(fopOutputLabelAna)
createDirIfNotExist(fopPerProject)

list_dir = os.listdir(fopDataset)   # Convert to lower case
list_dir =sorted(list_dir)


listSystemNames=[]
dictTextFrequenceAllProjects={}
index=0
for filename in list_dir:
    if not filename.endswith('.csv'):
        continue
    print(filename)

    index=index+1
    if index==5:
        break
    fpCsv=fopDataset+filename
    nameSystem=filename.replace('.csv','')
    listSystemNames.append(nameSystem)
    raw_data = pd.read_csv(fpCsv)
    columnId = raw_data['issuekey']
    columnRegStory = raw_data['storypoint']
    list_item_label=[]
    dictItemTextlbl={}


    listText=[]
    listLabel=[]

    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
        strContent = preprocess(strContent).replace('\t', ' ').replace('\n', ' ').replace(',', ' ').strip()
        intValue = int(columnRegStory[i])
        listText.append(strContent)
        listLabel.append(intValue)

    '''
    X_train_1, X_test, y_train_1, y_test = train_test_split(listText, listLabel, test_size=0.2,
                                                            shuffle=False,
                                                            stratify=None)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train_1, y_train_1, test_size=0.2,
                                                      shuffle=False,
                                                      stratify=None)
    '''
    dictTextFrequenceAllProjects[nameSystem]={}

    for i in range(0, len(listText)):
        strContent = listText[i]
        intValue = listLabel[i]
        if not intValue in dictTextFrequenceAllProjects[nameSystem].keys():
            dictTextFrequenceAllProjects[nameSystem][intValue]={}

        wordsList = nltk.word_tokenize(strContent)
        tagged = nltk.pos_tag(wordsList)
        # print('tagged {}'.format(type(tagged[0][0])))
        arrWords = []
        for it in tagged:
            strIt = '{} --- {}'.format(it[1], it[0]).replace(',','SCOLON')
            arrWords.append(strIt)
        #arrWords=word_tokenize(strContent)
        for j in range(0,len(arrWords)):
            wordItem=arrWords[j]
            dictItemTextlbl=dictTextFrequenceAllProjects[nameSystem][intValue]
            if wordItem in dictItemTextlbl.keys():
                dictItemTextlbl[wordItem]=dictItemTextlbl[wordItem]+1
            else:
                dictItemTextlbl[wordItem]=1
    for intValue in dictTextFrequenceAllProjects[nameSystem].keys():
        dictIt=dictTextFrequenceAllProjects[nameSystem][intValue]
        dictIt=dict( sorted(dictIt.items(), key=operator.itemgetter(1),reverse=True))

#listSystemNames=dictTextFrequenceAllProjects[nameSystem].keys()
numMax=100
for i in range(0,len(listSystemNames)):
    systemI = listSystemNames[i]
    arrNameI=systemI.split('_')
    nameAbI=arrNameI[2]
    fpExcelItem=fopPerProject+systemI+'.xlsx'
    print(fpExcelItem)
    writer = pd.ExcelWriter(fpExcelItem, engine='xlsxwriter')
    for j in range(0,len(listSystemNames)):
        if i==j:
            continue
        systemJ=listSystemNames[j]
        print('couple {} {}'.format(systemI, systemJ))
        arrNameJ = systemJ.split('_')
        nameAbJ=arrNameJ[2]
        dfInter,dfSubtract=extractOverlapAndMatchBetweenSystems(systemI,systemJ,dictTextFrequenceAllProjects,numMax)
        nameInter=nameAbI+'_'+nameAbJ+'_inter'
        nameSub = nameAbI + '_' + nameAbJ + '_sub'
        dfInter.to_excel(writer, sheet_name=nameInter, index=False)
        dfSubtract.to_excel(writer, sheet_name=nameSub, index=False)
    writer.save()
    print('finish excel of {}'.format(systemI))













