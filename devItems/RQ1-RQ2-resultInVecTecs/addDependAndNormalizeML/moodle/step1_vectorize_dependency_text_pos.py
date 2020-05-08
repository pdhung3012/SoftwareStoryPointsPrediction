from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import logging
from stanfordcorenlp import StanfordCoreNLP
from os import listdir
from os.path import isfile, join

fopVectorAllSystems='data/testVectorSystems_Tfidf4_depends_moodle/'
fopTextPreprocess='data/textPreprocess_Tfidf4_depends_moodle/'
fopDataset='../dataset/'

import stanza

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
    text = textInLine.replace('://',' ').replace('/',' ').replace('\'',' ').replace('\"',' ')
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

def getPCAFromTFIDFVector(listText,ngram,ncomps):
    # get vector using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram))
    X_old = vectorizer.fit_transform(listText)
    X_old = X_old.toarray()
    # print('len {}'.format(len(X_old)))
    # X = PCA().fit(X)
    pca = PCA(n_components=ncomps)
    X = pca.fit_transform(X_old)
    print('end vectorizer with length {}'.format(len(X_old[0])))
    return X



from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)
createDirIfNotExist(fopTextPreprocess)

fpHost='http://localhost'
# Debug the wrapper
# nlp = StanfordCoreNLP(r'path_or_host', logging_level=logging.DEBUG)

# Check more info from the CoreNLP Server
nlp = StanfordCoreNLP(fpHost,port=9000, logging_level=logging.DEBUG,  memory='8g')


for file in arrFiles:
    # if not file.endswith('csv'):
    #     continue
    if not file.endswith('moodle.csv'):
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

    titles = []
    descriptions = []
    # added_features=[]
    for i in range(0, len(raw_data['description'])):
        strTitle = str(raw_data['title'][i])
        strDesc = str(raw_data['description'][i])

        # lstItemFeatures=[]
        # lstItemFeatures.append(len(strTitle))
        # lstItemFeatures.append(len(strDesc))
        # simItem=getCosineDistance(strTitle,firstTokens(strDesc,len(strTitle)+5))
        # lstItemFeatures.append(simItem)
        # added_features.append(lstItemFeatures)
        # strContent = ' '.join([strTitle,' . ', strDesc])
        # strContent = ' '.join([str(raw_data['title'][i])])
        # strContent = ' '.join([strDesc])
        titles.append(strTitle)
        descriptions.append(strDesc)
        # titles_and_descriptions.append(str(strContent))

    Text1 = []
    Text2 = []
    Text3 = []
    Text4 = []
    Text5 = []
    Text6 = []

    listDependences = []
    index = 0
    for i in range(0, len(titles)):
        # lineStr=firstTokens(lineStr,50)
        lineStr1 = preprocess(titles[i])
        lineStr2 = preprocess(descriptions[i])
        # strToAdd = lineAppend
        strLem1 = 'Unknown'
        strDepend1 = 'Unknown'
        strPOS1 = 'Unknown'
        strLem2 = 'Unknown'
        strDepend2 = 'Unknown'
        strPOS2 = 'Unknown'
        try:
            strLem1 = lemmatization(nlp, lineStr1)
            strDepend1 = addDependenciesToSentenceDepend(nlp, lineStr1)
            strPOS1 = addDependenciesToSentencePOS(nlp, lineStr1)
            strLem2 = lemmatization(nlp, lineStr2)
            strDepend2 = addDependenciesToSentenceDepend(nlp, lineStr2)
            strPOS2 = addDependenciesToSentencePOS(nlp, lineStr2)
        except:
            print('{} error on issue {}'.format(index, columnId[index]))
        Text1.append(strLem1)
        Text2.append(strPOS1)
        Text3.append(strDepend1)
        Text4.append(strLem2)
        Text5.append(strPOS2)
        Text6.append(strDepend2)
        # text_after_tokenize.append(strToAdd)
        index = index + 1
        print('end {}'.format(index))
        # if index==102:
        #     break

    columnTitleRow = 'no,text\n'
    csv = open(fpTextInfo, 'w')
    csv.write(columnTitleRow)
    for i in range(0, len(Text1)):
        strItemTitle = Text1[i].replace(',', ' ')
        strItemDesc = Text4[i].replace(',', ' ')

        csv.write(','.join([str(i + 1), strItemTitle, Text2[i], Text3[i], strItemDesc, Text5[i], Text6[i]]))
        if (i < (len(Text1) - 1)):
            csv.write('\n')
    csv.close()
    # get vector using TF-IDF
    ng = 4
    nc = 50
    print(len(Text3))
    X1 = getPCAFromTFIDFVector(Text1, ng, nc)
    X2 = getPCAFromTFIDFVector(Text2, ng, nc)
    X3 = getPCAFromTFIDFVector(Text3, ng, nc)
    X4 = getPCAFromTFIDFVector(Text4, ng, nc)
    X5 = getPCAFromTFIDFVector(Text5, ng, nc)
    X6 = getPCAFromTFIDFVector(Text6, ng, nc)

    X = []
    for i in range(0, len(X1)):
        XI = np.append(X1[i], X2[i])
        XI = np.append(XI, X3[i])
        XI = np.append(XI, X4[i])
        XI = np.append(XI, X5[i])
        XI = np.append(XI, X6[i])
        X.append(XI)
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
    for i in range(0,len(titles)):
        # arrTokens = word_tokenize(str(text_after_tokenize[i]))
        # if not has_vector_representation(dictWordVectors, str(text_after_tokenize[i])):
        #     continue
        # # arrTokens = word_tokenize(str(text_after_tokenize[i]))
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

