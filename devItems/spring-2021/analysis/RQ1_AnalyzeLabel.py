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
from UtilFunctions import createDirIfNotExist

fopDataset='../../dataset/'
fopOutputLabelAna='../../../../dataPapers/analysisSEE/'
fopPerProject=fopOutputLabelAna+'perProjects/'
fpLabelPerProject=fopPerProject+'perProject_labels.xlsx'
fpTotalLabelFreq=fopOutputLabelAna+'total_frequenceLbl.xlsx'

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
writer = pd.ExcelWriter(fpLabelPerProject, engine='xlsxwriter')

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
    for i in range(0, len(raw_data['description'])):
        intValue = int(columnRegStory[i])
        list_item_label.append(intValue)
        list_total_label.append(intValue)
    minItem,maxItem,percentItem,dictItem=statisticOnLabels(list_item_label)
    strMinMax='{},{},{},{},{}'.format(index,nameSystem,minItem,maxItem,percentItem)
    lstLabelMinMax.append(strMinMax)

    lstItemLabelCount = ['No,SP,NumOfAppearance']
    index2 = 0
    for key in dictItem.keys():
        index2 = index2 + 1
        strItem = '{},{},{}'.format(index, key, dictItem[key])
        lstItemLabelCount.append(strItem)
    dfItem = pd.read_csv(io.StringIO('\n'.join(lstItemLabelCount) ), sep=",")
    # Write each dataframe to a different worksheet.
    dfItem.to_excel(writer, sheet_name=nameSystem)

minTotal,maxTotal,percentTotal,dictTotal=statisticOnLabels(list_total_label)

strMinMax='{},{},{},{},{}'.format(index+1,'total',minTotal,maxTotal,percentTotal)
lstLabelMinMax.append(strMinMax)

writer2 = pd.ExcelWriter(fpTotalLabelFreq, engine='xlsxwriter')
dfItem = pd.read_csv(io.StringIO('\n'.join(lstLabelMinMax )), sep=",")
dfItem.to_excel(writer2, sheet_name=nameSystem)
writer2.save()


lstTotalLabelCount=['No,SP,NumOfAppearance']
index=0
for key in dictTotal.keys():
    index=index+1
    strItem='{},{},{}'.format(index,key,dictTotal[key])
    lstTotalLabelCount.append(strItem)
dfAll = pd.read_csv(io.StringIO('\n'.join(lstTotalLabelCount) ), sep=",")
# Write each dataframe to a different worksheet.
dfAll.to_excel(writer, sheet_name='total')
writer.save()





