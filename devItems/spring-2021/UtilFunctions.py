import os
import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


from numpy import unique

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

def preprocessText(strInput,stop_words,ps,lemmatizer):
    word_tokens = word_tokenize(strInput)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    strTemp=' '.join(filtered_sentence)

    words=word_tokenize(strTemp)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)
    strOutput=strTemp
    return strOutput

def preprocessTextV2(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)
    strOutput=strTemp
    return strOutput


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


def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)



# Python program to illustrate the intersection
# of two lists in most simple way
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
def diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif