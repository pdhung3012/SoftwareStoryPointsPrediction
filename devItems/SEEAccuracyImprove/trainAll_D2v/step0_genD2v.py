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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# nameSystem='bamboo'
fopModel='model_d2v/'

# fopTextPreprocess='te'+nameSystem+'/'
fopDataset='../../dataset/'

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
createDirIfNotExist(fopModel)
# createDirIfNotExist(fopTextPreprocess)

fpHost='http://localhost'
# Debug the wrapper
# nlp = StanfordCoreNLP(r'path_or_host', logging_level=logging.DEBUG)

# Check more info from the CoreNLP Server
nlp = StanfordCoreNLP(fpHost,port=9000, logging_level=logging.DEBUG,  memory='8g')

Text1 = []
Text2 = []
Text3 = []
Text4 = []
Text5 = []
Text6 = []
ID=[]
StoryReg=[]
StoryClass=[]
Systems=[]
for file in arrFiles:
    if not file.endswith('csv'):
        continue
    # if not file.endswith(nameSystem+'.csv'):
    #     continue
    fileCsv = fopDataset + file
    # fpVectorItemCate=fopVectorAllSystems+file.replace('.csv','')+'_category.csv'
    # fpVectorItemReg = fopVectorAllSystems + file.replace('.csv','') + '_regression.csv'
    # fpTextInfo = fopVectorAllSystems + file.replace('.csv', '') + '_textInfo.csv'
    nameSystem=file.replace('.csv','')

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
        titles.append(strTitle)
        descriptions.append(strDesc)
        ID.append((str(columnId[i])))
        StoryReg.append(str(columnRegStory[i]))
        StoryClass.append(str(columnCateStory[i]))
        Systems.append(nameSystem)
        # titles_and_descriptions.append(str(strContent))



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
        print('end {} '.format(index))
        # if index==102:
        #     break
    print('Finish {}with length {}'.format(file,len(Text1)))
    # break

dictVocab={}
lstAll=[]
fpTextCorpus=fopModel+'text-16-project.csv'
file1=open(fpTextCorpus,'w')
file1.write('System,ID,StoryReg,StoryClass,Text1,Text2,Text3,Text4,Text5,Text6\n')
file1.close()
for index in range(0,len(Text1)):
    strTextIndex=','.join([Systems[index],ID[index],StoryReg[index],StoryClass[index],Text1[index].replace(',',' ').replace(';',' '),Text2[index].replace(',',' ').replace(';',' '),Text3[index].replace(',',' ').replace(';',' '),Text4[index].replace(',',' ').replace(';',' '),Text5[index].replace(',',' ').replace(';',' '),Text6[index].replace(',',' ').replace(';',' '),'\n'])
    file1 = open(fpTextCorpus, 'a')
    file1.write(strTextIndex)
    file1.close()


lstAll=Text1+Text2+Text3+Text4+Text5+Text6

tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(lstAll)]

max_epochs = 100
vec_size = 50
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
fpModelData=fopModel+'d2v_all.model'
model.save(fpModelData)
print("Model Saved")

