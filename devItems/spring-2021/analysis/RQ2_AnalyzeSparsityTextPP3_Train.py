from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
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
import sys
sys.path.append('../')
from UtilFunctions import createDirIfNotExist,preprocessTextV3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


fopDataset='../../dataset/'
fopOutputLabelAna='../../../../dataPapers/analysisSEE/'
fopPerProject=fopOutputLabelAna+'perProjects/'
fpRQ2PerProject= fopPerProject + 'rq2_pp3_textCount_train.xls'

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

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

list_total_count=[]
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


    listText=[]
    listLabel=[]

    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), ' ', str(raw_data['description'][i])])
        strContent = strContent.replace('\t', ' ').replace('\n', ' ').replace(',', ' ').strip()
        strContent=preprocessTextV3(strContent,ps,lemmatizer)
        intValue = int(columnRegStory[i])
        listText.append(strContent)
        listLabel.append(intValue)


    X_train, X_test, y_train, y_test = train_test_split(listText, listLabel, test_size=0.2,
                                                            shuffle=False,
                                                            stratify=None)

    # X_train, X_dev, y_train, y_dev = train_test_split(X_train_1, y_train_1, test_size=0.2,
    #                                                   shuffle=False,
    #                                                   stratify=None)

    for i in range(0, len(X_train)):
        strContent = X_train[i]
        intValue = y_train[i]

        wordsList = nltk.word_tokenize(strContent)
        arrWords=wordsList
        # tagged = nltk.pos_tag(wordsList)
        # # print('tagged {}'.format(type(tagged[0][0])))
        # arrWords = []
        # for it in tagged:
        #     strIt = '{} --- {}'.format(it[1], it[0])
        #     arrWords.append(strIt)
        #arrWords=word_tokenize(strContent)
        for j in range(0,len(arrWords)):
            wordItem=arrWords[j]
            if wordItem in dictItemTextlbl.keys():
                dictItemTextlbl[wordItem]=dictItemTextlbl[wordItem]+1

            else:
                dictItemTextlbl[wordItem]=1

            if wordItem in dictTotalTextAndLabel.keys():
                dictTotalTextAndLabel[wordItem]=dictTotalTextAndLabel[wordItem]+1
            else:
                dictTotalTextAndLabel[wordItem]=1


    lstItemLabelCount = ['No,Text,Count']
    index2 = 0
    for key in dictItemTextlbl.keys():
        index2 = index2 + 1
        count=dictItemTextlbl[key]
        strItem = '{},{},{}'.format(index2,key, count )
        lstItemLabelCount.append(strItem)
    #print('{}'.format('\n'.join(lstItemLabelCount)))
    dfItem = pd.read_csv(io.StringIO('\n'.join(lstItemLabelCount) ), sep=",")
    dfItem=dfItem.sort_values(by=['Count'], ascending=[True])
    # Write each dataframe to a different worksheet.
    dfItem.to_excel(writer, sheet_name=nameSystem, index=False)
    #break
#writer.save()


lstTotalLabelCount=['No,Text,Count']
index=0
for key in dictTotalTextAndLabel.keys():
    index=index+1
    count = dictTotalTextAndLabel[key]
    strItem = '{},{},{}'.format(index,key, count)
    lstTotalLabelCount.append(strItem)
#writer2 = pd.ExcelWriter(fpTotalRQ2, engine='xlsxwriter')
dfItem = pd.read_csv(io.StringIO('\n'.join(lstTotalLabelCount )), sep=",")
dfItem=dfItem.sort_values(by=['Count'], ascending=[True])
dfItem.to_excel(writer, sheet_name='total', index=False)
writer.save()

#dfAll = pd.read_csv(io.StringIO('\n'.join(lstTotalLabelCount) ), sep=",")
# Write each dataframe to a different worksheet.
#dfAll.to_excel(writer, sheet_name='total')





