from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#nltk.download()


import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from numpy import unique
from sklearn.metrics import mean_squared_error, mean_absolute_error

def initDefaultTextEnvi():
    nlp_model = spacy.load('en_core_web_sm')
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp_model,nlp

def getSentences(text,nlp):
    result=None
    try:
        document = nlp(text)
        result= [sent.string.strip() for sent in document.sents]
    except Exception as e:
        print('sone error occured {}'.format(str(e)))
    return result

'''
def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]
'''

def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

from UtilFunctions import createDirIfNotExist

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
#        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    #print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def processSentence(sentence,nlp_model):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)


from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == "__main__":

    fopDataset = '../dataset/'
    fopFatherFolder = '../../../dataPapers/dataTextLevelPaper/'

    fopRoot = '/home/hungphd/git/dataPapers/dataTextGCN/'

    list_dir = os.listdir(fopDataset)  # Convert to lower case
    list_dir = sorted(list_dir)
    fpResultCosine = fopFatherFolder + "/resultCosine.txt"
    fff=open(fpResultCosine,'w')
    fff.write('')
    fff.close()

    for filename in list_dir:
        if not filename.endswith('.csv'):
            continue
        fnSystem = filename
        print('System {}'.format(fnSystem))
        fnSystemAbbrev = filename.replace('.csv', '')
        fopOutputDs = fopFatherFolder+fnSystemAbbrev+'/'
        fpOutputTextIndex = fopFatherFolder+fnSystemAbbrev+'.txt'
        fpOutputTextTrainIndex = fopFatherFolder + fnSystemAbbrev + '.train.txt'
        fpOutputTestLbl= fopFatherFolder + fnSystemAbbrev + '_testLblStep1.txt'
        fopOutputCosineApp = fopFatherFolder + fnSystemAbbrev + '/cosineDistance/'
        fpPred = fopOutputCosineApp + "/test_pred.txt"
        fpCompareDetails = fopOutputCosineApp + "/compareDetails.txt"

        fpOutputPercentLbl = fopOutputDs + 'percentLabel.txt'
        fopOutputLabelInfo = fopOutputDs + 'labelInfo/'
        fopOutputLabelVocab = fopOutputDs + 'labelVocab/'
        fopRoot=fopFatherFolder

        createDirIfNotExist(fopOutputDs)
        createDirIfNotExist(fopOutputLabelInfo)
        createDirIfNotExist(fopOutputLabelVocab)
        createDirIfNotExist(fopOutputCosineApp)

        fnSystem=fnSystemAbbrev+'.csv'
        fileCsv = fopDataset + fnSystem

        raw_data = pd.read_csv(fileCsv)
        raw_data_2 = pd.read_csv(fileCsv)
        columnId = raw_data['issuekey']
        columnRegStory = raw_data_2['storypoint']
        titles_and_descriptions = []
        colTest=[]
        for i in range(0, len(raw_data['description'])):
            strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
            strContent=preprocess(strContent).replace('\t',' ').replace('\n',' ').strip()

            #fopOutputCosineApp

            titles_and_descriptions.append(str(strContent))
            colTest.append(int(columnRegStory[i]))



        X_train, X_test, y_train, y_test = train_test_split(titles_and_descriptions, colTest, test_size=0.2, shuffle=False,
                                                            stratify=None)
        #input('sss ')
        lstCosineResult=[]
        lstCompareDetails=[]
        for indexTest in range(0,len(X_test)):
            itemTest=str(X_test[indexTest])
            labelTest=y_test[indexTest]
            listTrainSimScore=[]
            for indexTrain in range(0,len(X_train)):
                itemTrain = str(X_train[indexTrain])
                labelTrain = y_train[indexTrain]
                #arrScore=get_cosine_sim(itemTrain,itemTest)
                scoreIt=get_jaccard_sim(itemTrain,itemTest)
                #print('score {}'.format(scoreIt))
               # listTrainSimScore.append(0.7)
                listTrainSimScore.append(scoreIt)

            maxS=max(listTrainSimScore)
            indexWin=listTrainSimScore.index(maxS)
            labelSelected=y_train[indexWin]
            itemSelected=str(X_train[indexWin])
            strCompare='{}\texpected\t{}\t{}\t{}\n{}\texpected\t{}\t{}\t{}'.format(indexTest,labelTest,maxS,itemTest,indexTest,labelSelected,maxS,itemSelected)
            lstCompareDetails.append(strCompare)
            lstCosineResult.append(labelSelected)
            print('{}\t{}'.format(indexTest,len(X_test)))


        #input('I love you ')
        maeAccuracy= mean_absolute_error(y_test , lstCosineResult)

        fff=open(fpPred,'w')
        lstStr=[]
        for id2 in range(0,len(y_test)):
            lstStr.append('{}\t{}'.format(y_test[id2],lstCosineResult[id2]))
        fff.write('\n'.join(lstStr))
        fff.close()

        fff = open(fpCompareDetails, 'w')
        fff.write('\n'.join(lstCompareDetails))
        fff.close()

        fff = open(fpResultCosine, 'a')
        fff.write('{}\t{}\n'.format(fnSystemAbbrev,maeAccuracy))
        fff.close()
        print('done {}'.format(fnSystemAbbrev))















print('Done')




