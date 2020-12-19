from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

fopVectorAllSystems='data/testVectorSystems_Tfidf4_depends_xd/'
fopTextPreprocess='data/textPreprocess_Tfidf4_depends_xd/'
fopDataset='../dataset/'
import Levenshtein

import stanza

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

# def addDependenciesToSentencePOS(docObj):
#     lstSentences=docObj.sentences
#     lstOutput=[]
#     for sen in lstSentences:
#         words=sen._words
#         lstDepInfo=[]
#         # depends=dict(depends)
#         for w in words:
#             strElement=' '.join([w.upos])
#             lstDepInfo.append(strElement)
#         strDep=' '.join(lstDepInfo)
#         lstOutput.append((strDep))
#     strResult=' '.join(lstOutput)
#     return strResult
def addDependenciesToSentencePOS(nlp,strObj):
    lstTuples=nlp.pos_tag(strObj)
    lstOutput=[]
    for sen in lstTuples:
        strDep=sen[1]
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

def addDependenciesToSentenceDepend(nlp,strObj):
    lstTuples=nlp.dependency_parse(strObj)
    lstOutput=[]
    for sen in lstTuples:
        strDep=sen[0]
        lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult

import json
def lemmatization(nlp,strObj):
    sentences = nlp.annotate(strObj, properties={'annotators': 'lemma'})
    jsonObj = json.loads(sentences)
    lstOutput=[]
    for sen in jsonObj['sentences']:
        for token in sen['tokens']:
            strDep=str(token['lemma'])
            lstOutput.append((strDep))
    strResult=' '.join(lstOutput)
    return strResult
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def getCosineDistance(str1,str2):
    score=0
    corpus = [str1, str2]
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(corpus)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['Document 0', 'Document 1'])
    simArr=cosine_similarity(trsfm[0:1], trsfm)
    score=simArr[0][1]
    return score

def firstTokens(textInLine,numTokens):
    lstIn=word_tokenize(textInLine)
    lstOut=[]
    for i in range(0,min(len(lstIn),numTokens)):
        lstOut.append(lstIn[i])
    strResult = ' '.join(lstOut)
    return strResult

def preprocess(textInLine):
    text = textInLine.lower().replace('://',' ').replace('/',' ').replace('\'',' ').replace('\"',' ')
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")



import logging
from stanfordcorenlp import StanfordCoreNLP
from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)
createDirIfNotExist(fopTextPreprocess)

# nlp = stanza.Pipeline() # This sets up a default neural pipeline in English
fpHost='http://localhost'
# Debug the wrapper
# nlp = StanfordCoreNLP(r'path_or_host', logging_level=logging.DEBUG)

# Check more info from the CoreNLP Server
nlp = StanfordCoreNLP(fpHost,port=9000, logging_level=logging.DEBUG,  memory='8g')


for file in arrFiles:
    # if not file.endswith('csv'):
    #     continue
    if not file.endswith('springxd.csv'):
        continue
    fileCsv = fopDataset + file
    fpVectorItemCate=fopVectorAllSystems+file.replace('.csv','')+'_category.csv'
    fpVectorItemReg = fopVectorAllSystems + file.replace('.csv','') + '_regression.csv'
    fpTextInfo = fopTextPreprocess + file.replace('.csv', '') + '_textInfo.csv'

    raw_data = pd.read_csv(fileCsv)
    raw_data_2 = pd.read_csv(fileCsv)
    columnId=raw_data['issuekey']
    columnRegStory=raw_data_2['storypoint']
    raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0  # small
    raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), 'storypoint'] = 1  # medium
    raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), 'storypoint'] = 2  # large
    raw_data.loc[raw_data.storypoint > 15, 'storypoint'] = 3  # very large
    columnCateStory = raw_data['storypoint']

    titles_and_descriptions = []
    added_features=[]
    for i in range(0, len(raw_data['description'])):
        strTitle=str(raw_data['title'][i])
        strDesc=str(raw_data['description'][i])
        lstItemFeatures=[]
        lstItemFeatures.append(len(strTitle))
        lstItemFeatures.append(len(strDesc))
        simItem=getCosineDistance(strTitle,firstTokens(strDesc,len(strTitle)+5))
        lstItemFeatures.append(simItem)
        added_features.append(lstItemFeatures)


        strContent = ' '.join([strTitle,' . ', strDesc])
        # strContent = ' '.join([str(raw_data['title'][i])])
        # strContent = ' '.join([strDesc])
        titles_and_descriptions.append(str(strContent))

    text_after_tokenize = []
    listDependences=[]
    index=0
    for lineStr in titles_and_descriptions:
        # lineStr=firstTokens(lineStr,50)
        lineAppend = preprocess(lineStr)

        strToAdd = lineAppend
        # try:
        #     # lineAppend =lemmatization(nlp,lineAppend)
        #     # doc = nlp(lineStr)
        #     strDepend = addDependenciesToSentenceDepend(nlp,lineStr)
        #     # strToAdd = ' '.join([lineAppend, strDepend])
        #     strToAdd = ' '.join([strDepend])
        # except:
        #     print('{} error on issue {}'.format(index,columnId[index]))
        text_after_tokenize.append(strToAdd)
        index=index+1
        print('end {}'.format(index))
        # if index==102:
        #     break

    columnTitleRow='no,text\n'
    csv = open(fpTextInfo, 'w')
    csv.write(columnTitleRow)
    for i in range(0, len(text_after_tokenize)):
        strItem=text_after_tokenize[i].replace(',',' ')
        csv.write(','.join([str(i+1),strItem]))
        if(i<(len(text_after_tokenize)-1)):
            csv.write('\n')
    csv.close()
    # get vector using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_old = vectorizer.fit_transform(text_after_tokenize)
    X_old = X_old.toarray()

    # X = PCA().fit(X)
    pca = PCA(n_components=100)
    X_old=pca.fit_transform(X_old)
    # srp=GaussianRandomProjection(n_components=3)
    # X=srp.fit_transform(X)
    print('end vectorize')

    # add features
    # oldLen=len(X_old[0])
    # X=[]
    # for i in range(0,len(X_old)):
    #     XI = np.append(added_features[i],X_old[i])
    #     # XI = added_features[i]
    #     X.append(XI)

    X=X_old
    lenVectorOfWord = len(X[0])


    columnTitleRow = "no,story,"
    for i in range(0,lenVectorOfWord):
        item='feature-'+str(i+1)
        columnTitleRow = ''.join([columnTitleRow, item])
        if i!=lenVectorOfWord-1:
            columnTitleRow = ''.join([columnTitleRow,  ","])
    columnTitleRow = ''.join([columnTitleRow, "\n"])
    csv = open(fpVectorItemCate, 'w')
    csv.write(columnTitleRow)

    csv2 = open(fpVectorItemReg, 'w')
    csv2.write(columnTitleRow)



    corpusVector = []
    for i in range(0,len(text_after_tokenize)):
        # arrTokens = word_tokenize(str(text_after_tokenize[i]))
        # if not has_vector_representation(dictWordVectors, str(text_after_tokenize[i])):
        #     continue
        # # arrTokens = word_tokenize(str(text_after_tokenize[i]))
        # if columnRegStory[i]>30:
        #     continue
        vector= X[i]
        corpusVector.append(vector)
        # strVector=','.join(vector)
        strCate=str(columnCateStory[i])
        strReg=str(columnRegStory[i])
        # strRow=''.join([str(i+1),',','S-'+str(columnStoryPoints[i]),])
        # strRow = ''.join([str(i + 1), ',', 'S-' + strCate, ])
        strRow = ''.join([str(i + 1), ',', '' + strCate, ])
        strRow2 = ''.join([str(i + 1), ',', '' + strReg, ])
        for j in range(0,lenVectorOfWord):
            strRow=''.join([strRow,',',str(vector[j])])
            strRow2 = ''.join([strRow2, ',', str(vector[j])])
        strRow = ''.join([strRow, '\n'])
        strRow2 = ''.join([strRow2, '\n'])
        csv.write(strRow)
        csv2.write(strRow2)
    csv.close()
    csv2.close()
    print('Finish {}'.format(file))

