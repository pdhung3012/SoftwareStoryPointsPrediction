

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut

import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import nltk
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold, StratifiedKFold

import os;
from nltk.parse import CoreNLPParser
from sklearn.decomposition import PCA

def categorize(score):
    result=0
    if score <=1:
        result=0
    elif score>1 and score<=4:
        result=1
    else:
        result=2
    # return score
    return result

# Averaging Word Embeddings
def document_vector(strContent,dictWordVectors,lenVector):
    # remove out-of-vocabulary words
    arrWord=strContent.split(' ')
    listVectorWords=[]
    for word in arrWord:
        if word in dictWordVectors:
            arr=dictWordVectors[word]
            listVectorWords.append(arr)

    if len(listVectorWords)==0:
        arrResult=[]
        for i in range(0,lenVector):
            arrResult.append(0)
        return arrResult
    return np.mean(listVectorWords, axis=0)


# Load Data
url = 'https://raw.githubusercontent.com/SEAnalytics/datasets/master/storypoint/IEEE%20TSE2018/dataset/mulestudio.csv'
fpVector="data/vectorGlobeRegression.csv"
fpInputTrainedGloveVector="data/globe_words.txt"
raw_data = pd.read_csv(url)
raw_data.columns
# raw_data.head(6)
columnDescription=raw_data.description
columnStoryPoints=raw_data.storypoint

dictWordVectors={}
fRead=open(fpInputTrainedGloveVector,'r')
strVectorContent=fRead.read()
fRead.close()
lstContent=strVectorContent.split('\n')
lenVectorOfWord=0
for i in range(0,len(lstContent)):
    lstVectorItem=[]
    word=''
    arrItem=lstContent[i].split(' ')
    # print(str(arrItem))
    if len(arrItem)>2:
        word=arrItem[0]
        if lenVectorOfWord==0:
            lenVectorOfWord=len(arrItem)-1
        for j in range(1,len(arrItem)):
            lstVectorItem.append(float(arrItem[j]) )
        dictWordVectors[word]=lstVectorItem

columnTitleRow = "no,story,"
for i in range(0,lenVectorOfWord):
    item='feature-'+str(i+1)
    columnTitleRow = ''.join([columnTitleRow, item])
    if i!=lenVectorOfWord-1:
        columnTitleRow = ''.join([columnTitleRow,  ","])
columnTitleRow = ''.join([columnTitleRow, "\n"])
csv = open(fpVector, 'w')
csv.write(columnTitleRow)

corpusVector = []
for i in range(0,len(columnDescription)):
    vector=document_vector(str(columnDescription[i]),dictWordVectors,lenVectorOfWord)
    corpusVector.append(vector)
    # strVector=','.join(vector)
    strCate=str(categorize(columnStoryPoints[i]))
    # strRow=''.join([str(i+1),',','S-'+str(columnStoryPoints[i]),])
    # strRow = ''.join([str(i + 1), ',', 'S-' + strCate, ])
    strRow = ''.join([str(i + 1), ',', '' + strCate, ])
    for j in range(0,lenVectorOfWord):
        strRow=''.join([strRow,',',str(vector[j])])
    strRow = ''.join([strRow, '\n'])
    csv.write(strRow)
csv.close()



# vectorizer = TfidfVectorizer(ngram_range=(1, 1))
# X = vectorizer.fit_transform(corpusTrain)
# X = X.toarray()
#
# # X = PCA().fit(X)
# pca=PCA(n_components=100)
# aaa=pca.fit_transform(X)
# print(str(len(aaa[0])))
# lenVector=len(X[0])
# print(str(lenVector))
# columnTitleRow = "no,story,"
# for i in range(0,lenVector):
#     item='feature-'+str(i+1)
#     columnTitleRow = ''.join([columnTitleRow, item])
#     if i!=lenVector-1:
#         columnTitleRow = ''.join([columnTitleRow,  ","])
# columnTitleRow = ''.join([columnTitleRow, "\n"])
#
#
# csv = open(fpVector, 'w')
# csv.write(columnTitleRow)
# for i in range(0, len(corpusTrain)):
#     vectori = X[i]
#     print(str(i)+'_'+str(vectori[227]))
#     row=''.join([str(i+1),',','S-',str(columnStoryPoints[i]),','])
#     strVector=",".join(str(j) for j in vectori)
#     row = ''.join([row,strVector, "\n"])
#     # for j in range(0,len(vectori)):
#     #     row=''.join([row,',',str(vectori[j])])
#     #
#     # row = ''.join([row, "\n"])
#     csv.write(row)
# csv.close()






