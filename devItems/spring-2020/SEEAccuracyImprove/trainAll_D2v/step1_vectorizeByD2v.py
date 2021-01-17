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
fpModelD2v=fopModel+'d2v_all.model'
fpTextInfo=fopModel+'text-16-project.csv'
fpVecctorD2v=fopModel+'vector-16-project.csv'

# fopTextPreprocess='te'+nameSystem+'/'
fopDataset='../../dataset/'



def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")




createDirIfNotExist(fopModel)
# raw_data = pd.read_csv(fpTextInfo)

file=open(fpTextInfo,'r')
arrLines=file.read().split('\n')

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

for i in range(0,len(arrLines)):
    arrLineItem=arrLines[i].split(',')
    print(arrLines[i])
    if(len(arrLineItem)<=2):
        continue
    Text1.append(arrLineItem[4])
    Text2.append(arrLineItem[5])
    Text3.append(arrLineItem[6])
    Text4.append(arrLineItem[7])
    Text5.append(arrLineItem[8])
    Text6.append(arrLineItem[9])
    ID.append(arrLineItem[1])
    StoryReg.append(arrLineItem[2])
    StoryClass.append(arrLineItem[3])
    Systems.append(arrLineItem[0])

# Text1 = raw_data['Text1']
# Text2 = raw_data['Text2']
# Text3 = raw_data['Text3']
# Text4 = raw_data['Text4']
# Text5 = raw_data['Text5']
# Text6 = raw_data['Text6']
# ID=raw_data['ID']
# StoryReg=raw_data['StoryReg']
# StoryClass=raw_data['StoryClass']
# Systems=raw_data['System']

from gensim.models.doc2vec import Doc2Vec
model= Doc2Vec.load(fpModelD2v)
lenVectorOfWord =0
for i in range(0,len(Text1)):
    arrText1=word_tokenize(str(Text1[i]))
    arrText2 = word_tokenize(str(Text2[i]))
    arrText3 = word_tokenize(str(Text3[i]))
    arrText4 = word_tokenize(str(Text4[i]))
    arrText5 = word_tokenize(str(Text5[i]))
    arrText6 = word_tokenize(str(Text6[i]))
    print(arrText1)
    print(arrText2)
    print(arrText3)
    print(arrText4)
    print(arrText5)
    print(arrText6)


    vector1=model.infer_vector(arrText1)
    vector2 = model.infer_vector(arrText2)
    vector3 = model.infer_vector(arrText3)
    vector4 = model.infer_vector(arrText4)
    vector5 = model.infer_vector(arrText5)
    vector6 = model.infer_vector(arrText6)

    XI = np.append(vector1, vector2)
    XI = np.append(XI, vector3)
    XI = np.append(XI, vector4)
    XI = np.append(XI, vector5)
    XI = np.append(XI, vector6)
    if i==0:
        lenVectorOfWord = len(XI)
        columnTitleRow = "Systems,ID,StoryReg,StoryClass,"
        for i in range(0, lenVectorOfWord):
            item = 'feature-' + str(i + 1)
            columnTitleRow = ''.join([columnTitleRow, item])
            if i != lenVectorOfWord - 1:
                columnTitleRow = ''.join([columnTitleRow, ","])
        columnTitleRow = ''.join([columnTitleRow, "\n"])
        csv = open(fpVecctorD2v, 'w')
        csv.write(columnTitleRow)

    strRow = ''.join([str(Systems[i]), ',', str(ID[i]), ',', str(StoryReg[i]), ',', str(StoryClass[i]) ])
    for j in range(0, lenVectorOfWord):
        strRow = ''.join([strRow, ',', str(XI[j])])
    strRow = ''.join([strRow, '\n'])
    csv.write(strRow)
    print('{} finish'.format(i))


csv.close()


