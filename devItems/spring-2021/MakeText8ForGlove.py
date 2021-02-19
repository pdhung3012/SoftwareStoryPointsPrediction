from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#nltk.download()


fopDataset='../dataset/'
fopPretrain='../PretrainData/'
fpText8='../../../dataPapers/text8'

def addDependenciesToSentence(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        depends=sen._dependencies
        lstDepInfo=[]
        # depends=dict(depends)
        for deKey in depends:
            strElement=' '.join([deKey[2].text,deKey[0].text,deKey[1]])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

def addDependenciesToSentenceCompact(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        depends=sen._dependencies
        lstDepInfo=[]
        # depends=dict(depends)
        for deKey in depends:
            strElement=' '.join([deKey[1]])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

def addDependenciesToSentencePOS(docObj):
    lstSentences=docObj.sentences
    lstOutput=[]
    for sen in lstSentences:
        words=sen._words
        lstDepInfo=[]
        # depends=dict(depends)
        for w in words:
            strElement=' '.join([w.upos])
            lstDepInfo.append(strElement)
        strDep=' '.join(lstDepInfo)
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult


def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

from UtilFunctions import createDirIfNotExist



from os import listdir
from os.path import isfile, join

arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
list_dir = os.listdir(fopDataset)   # Convert to lower case
list_dir =sorted(list_dir)
print(str(list_dir))

listTotalStr=[]

for filename in list_dir:
    if not filename.endswith('.csv'):
        continue
    #if not file.endswith('moodle.csv'):
    #    continue
    fileCsv = fopDataset + filename
   # fpVectorItemCate=fopVectorAllSystems+filename.replace('.csv','')+'_category.csv'
    raw_data = pd.read_csv(fileCsv)
    raw_data_2 = pd.read_csv(fileCsv)
    columnId=raw_data['issuekey']
    columnRegStory=raw_data_2['storypoint']

    for i in range(0, len(raw_data['description'])):
        lenItem=len(raw_data['description'])
        strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
        strContent = preprocess(strContent).replace('\t', ' ').replace('\n', ' ').replace(',', ' ').strip()
        listTotalStr.append(strContent)
        wordsList = nltk.word_tokenize(strContent)
        tagged = nltk.pos_tag(wordsList)
        # print('tagged {}'.format(type(tagged[0][0])))
        lstContentI = []
        for it in tagged:
            strIt = '{} {}'.format(it[0], it[1])
            lstContentI.append(strIt)
        strContentPos = ' '.join(lstContentI)
        listTotalStr.append(strContentPos)
    print('name {}'.format(filename))
    #break

list_dir = os.listdir(fopPretrain)   # Convert to lower case
list_dir =sorted(list_dir)

print(str(list_dir))
for filename in list_dir:
    if not filename.endswith('.csv'):
        continue

    fileCsv = fopPretrain + filename
   # fpVectorItemCate=fopVectorAllSystems+filename.replace('.csv','')+'_category.csv'
    raw_data = pd.read_csv(fileCsv)

    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
        strContent = preprocess(strContent).replace('\t', ' ').replace('\n', ' ').replace(',', ' ').strip()
        listTotalStr.append(strContent)
        '''
        wordsList = nltk.word_tokenize(strContent)
        tagged = nltk.pos_tag(wordsList)
        # print('tagged {}'.format(type(tagged[0][0])))
        lstContentI = []
        for it in tagged:
            strIt = '{} {}'.format(it[0], it[1])
            lstContentI.append(strIt)
        strContentPos = ' '.join(lstContentI)
        listTotalStr.append(strContentPos)
        '''
    print('name {}'.format(filename))
    #break

fff=open(fpText8,'w')
fff.write('\n'.join(listTotalStr))
fff.close()



