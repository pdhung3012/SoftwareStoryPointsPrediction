from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split



import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt


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

def returnIDOrGenNewID(content,dictVocab):
    result=-1
    if content in dictVocab.keys():
        result=dictVocab[content]
    else:
        result=len(dictVocab.keys())+1
        dictVocab[content]=result
    return result

def extractGraphKeyFromTriple(triples,dictVocab):
    G = nx.Graph()
    for triple in triples:
#        print('aaaa {}'.format(triple[0]))
        key0 = returnIDOrGenNewID(triple[0], dictVocab)
        key1 = returnIDOrGenNewID(triple[1], dictVocab)
        key2 = returnIDOrGenNewID(triple[2], dictVocab)
        G.add_node(key0)
        G.add_node(key1)
        G.add_node(key2)
        G.add_edge(key0, key1)
        G.add_edge(key1, key2)
    return G


def extractGraphSpekFromText(content,label,dictVocab,nlp_model,nlp):
    g=None
    sentences = getSentences(content, nlp)
    # nlp_model = spacy.load('en_core_web_sm')

    triples = []
    for sentence in sentences:
        triples.append(processSentence(sentence, nlp_model))
    g = extractGraphKeyFromTriple(triples, dictVocab)
    adjacency_matrix = nx.adjacency_matrix(g)
    graphSpek = Graph()
    graphSpek.a = adjacency_matrix
    graphSpek.y=label
    return graphSpek

if __name__ == "__main__":

    fopDataset = '../../dataset/'
    fopOutputDs = '../../../../dataPapers/dataTextGCN/msfm/'
    fpOutputTextIndex = '../../../../dataPapers/dataTextGCN/msfm.txt'
    fnSystem='mulestudio.csv'
    fileCsv = fopDataset + fnSystem

    raw_data = pd.read_csv(fileCsv)
    raw_data_2 = pd.read_csv(fileCsv)
    columnId = raw_data['issuekey']
    columnRegStory = raw_data_2['storypoint']
    titles_and_descriptions = []
    colTest=[]
    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
        strContent=preprocess(strContent)
        titles_and_descriptions.append(str(strContent))
        colTest.append(str(columnRegStory[i]))

    X_train, X_test, y_train, y_test = train_test_split(titles_and_descriptions, colTest, test_size=0.2, shuffle=False,
                                                        stratify=None)
    #print('y test{}'.format(y_test))

    createDirIfNotExist(fopOutputDs)
    fopOutTrain=fopOutputDs+'train/'
    fopOutTest = fopOutputDs + 'test/'
    createDirIfNotExist(fopOutTrain)
    createDirIfNotExist(fopOutTest)

    listIndexStr=[]

    for i in range(0,len(X_test)):
        strDocId=str(i+1)
        strLbl=y_test[i]
        #print('test lbl '.format(strLbl))
        fopItem=fopOutTest+strLbl+"/"
        createDirIfNotExist(fopItem)
        fff=open(fopItem+strDocId,'w')
        fff.write(X_test[i])
        fff.close()
        strLine='/home/hungphd/git/dataPapers/dataTextGCN/msfm/test/'+strDocId+'\ttest\t'+strLbl
        listIndexStr.append(strLine)

    for i in range(0,len(X_train)):
        strDocId=str(i+1)
        strLbl=str(y_train[i])
        fopItem=fopOutTrain+strLbl+"/"
        createDirIfNotExist(fopItem)
        fff=open(fopItem+strDocId,'w')
        fff.write(X_train[i])
        fff.close()
        strLine='/home/hungphd/git/dataPapers/dataTextGCN/msfm/train/'+strDocId+'\ttrain\t'+strLbl
        listIndexStr.append(strLine)

fff=open(fpOutputTextIndex,'w')
fff.write('\n'.join(listIndexStr))
fff.close()


print('Done')




