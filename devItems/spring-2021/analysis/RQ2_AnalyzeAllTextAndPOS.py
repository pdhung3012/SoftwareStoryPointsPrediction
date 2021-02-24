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
from UtilFunctions import createDirIfNotExist,preprocess

fopDataset='../../dataset/'
fopOutputLabelAna='../../../../dataPapers/analysisSEE/'
fopPerProject=fopOutputLabelAna+'perProjects/'
fpRQ2PerProject= fopPerProject + 'perProject_rq2_pos.xlsx'
fpTotalRQ2= fopOutputLabelAna + 'total_rq2.xlsx'

def statisticOnLabels(listLabels):
    percentage=0.0
    minVal=min(listLabels)
    maxVal=max(listLabels)
    lstUnq=unique(listLabels)
    percentage=len(lstUnq)*1.0/(maxVal-minVal+1)
    dictLbl={}
    for item in lstUnq:
        dictLbl[item]=listLabels.count(item)
    sorted_d = dict(sorted(dictLbl.items(), key=operator.itemgetter(1), reverse=True))
    return minVal,maxVal,percentage,sorted_d

createDirIfNotExist(fopOutputLabelAna)
createDirIfNotExist(fopPerProject)

list_dir = os.listdir(fopDataset)   # Convert to lower case
list_dir =sorted(list_dir)

list_total_label=[]
lstLabelMinMax=['No,Project,MinSp,MaxSp,PercentageOverlap']
index=0
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(fpRQ2PerProject, engine='xlsxwriter')

dictTotalTextAndLabel={}
for filename in list_dir:
    if not filename.endswith('.csv'):
        continue
    print(filename)
    index=index+1
    fpCsv=fopDataset+filename
    nameSystem=filename.replace('.csv','')
    raw_data = pd.read_csv(fpCsv)
    columnId = raw_data['issuekey']
    columnRegStory = raw_data['storypoint']
    list_item_label=[]
    dictItemTextlbl={}
    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
        strContent = preprocess(strContent).replace('\t', ' ').replace('\n', ' ').replace(',', ' ').strip()
        intValue = int(columnRegStory[i])

        wordsList = nltk.word_tokenize(strContent)
        tagged = nltk.pos_tag(wordsList)
        # print('tagged {}'.format(type(tagged[0][0])))
        arrWords = []
        for it in tagged:
            strIt = '{} --- {}'.format(it[1], it[0])
            arrWords.append(strIt)
        #arrWords=word_tokenize(strContent)
        for j in range(0,len(arrWords)):
            wordItem=arrWords[j]
            if wordItem in dictItemTextlbl.keys():
                lst=dictItemTextlbl[wordItem]
                lst.append(intValue)
            else:
                lst=[]
                lst.append(intValue)
                dictItemTextlbl[wordItem]=lst

            if wordItem in dictTotalTextAndLabel.keys():
                lst=dictTotalTextAndLabel[wordItem]
                lst.append(intValue)
            else:
                lst=[]
                lst.append(intValue)
                dictTotalTextAndLabel[wordItem]=lst


    lstItemLabelCount = ['No,Text,NumOfUniqueLabel,NumOfRecordsAppeared,UniqueLabels']
    index2 = 0
    for key in dictItemTextlbl.keys():
        index2 = index2 + 1
        lst=dictItemTextlbl[key]
        lenApp=len(lst)
        setLbl=sorted(unique(lst))
        strLbl=' - '.join(map(str,setLbl))
        strItem = '{},{},{},{},{}'.format(index2,key, len(setLbl),lenApp, strLbl )
        lstItemLabelCount.append(strItem)
    #print('{}'.format('\n'.join(lstItemLabelCount)))
    dfItem = pd.read_csv(io.StringIO('\n'.join(lstItemLabelCount) ), sep=",")
    dfItem=dfItem.sort_values(by=['NumOfUniqueLabel','NumOfRecordsAppeared'], ascending=[True,False])
    # Write each dataframe to a different worksheet.
    dfItem.to_excel(writer, sheet_name=nameSystem, index=False)
    #break
#writer.save()


lstTotalLabelCount=['No,Text,NumOfUniqueLabel,NumOfRecordsAppeared,UniqueLabels']
index=0
for key in dictTotalTextAndLabel.keys():
    index=index+1
    lst = dictTotalTextAndLabel[key]
    lenApp = len(lst)
    setLbl = sorted(unique(lst))
    strLbl = ' - '.join(map(str, setLbl))
    strItem = '{},{},{},{},{}'.format(index,key, len(setLbl),lenApp, strLbl)
    lstTotalLabelCount.append(strItem)
#writer2 = pd.ExcelWriter(fpTotalRQ2, engine='xlsxwriter')
dfItem = pd.read_csv(io.StringIO('\n'.join(lstTotalLabelCount )), sep=",")
dfItem=dfItem.sort_values(by=['NumOfUniqueLabel','NumOfRecordsAppeared'], ascending=[True,False])
dfItem.to_excel(writer, sheet_name='total', index=False)
writer.save()

#dfAll = pd.read_csv(io.StringIO('\n'.join(lstTotalLabelCount) ), sep=",")
# Write each dataframe to a different worksheet.
#dfAll.to_excel(writer, sheet_name='total')





