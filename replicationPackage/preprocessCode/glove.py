
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim


fopVectorAllSystems='../result/'
fopDataset='../data/datasetFromTSE2018/'
print('Remember to download the google pretrained model glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/ and extract it inside data folder')

fpInputTrainedGloveVector="../data/glove.6B.50d.txt"



def preprocess(textInLine):
    text = textInLine.lower()
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



def has_vector_representation(dictWordVectors, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    arrWord = strContent.split(' ')
    listVectorWords = []
    for word in arrWord:
        if word in dictWordVectors:
            arr = dictWordVectors[word]
            listVectorWords.append(arr)
    if len(listVectorWords) == 0:
        return False
    return True
    # return not all(word not in glove_dict.keys for word in doc)



# # Averaging Word Embeddings
# def document_vector(strContent,dictWordVectors,lenVector):
#     # remove out-of-vocabulary words
#     arrWord=strContent.split(' ')
#     listVectorWords=[]
#     for word in arrWord:
#         if word in dictWordVectors:
#             arr=dictWordVectors[word]
#             listVectorWords.append(arr)
#
#     if len(listVectorWords)==0:
#         arrResult=[]
#         for i in range(0,lenVector):
#             arrResult.append(0)
#         return arrResult
#     return np.mean(listVectorWords, axis=0)

# Averaging Word Embeddings
def document_vector_glove(strContent,dictWordVectors):
    # remove out-of-vocabulary words
    # doc = [word for word in doc if word in dictWordVectors.keys()]
    arrWord=strContent.split(' ')
    listVectorWords=[]
    for word in arrWord:
        if word in dictWordVectors:
            arr=dictWordVectors[word]
            listVectorWords.append(arr)

    return np.mean(listVectorWords, axis=0)


from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)

dictWordVectors={}
# fRead=open(fpInputTrainedGloveVector,'r')
# strVectorContent=fRead.read()
# fRead.close()
lstContent=[]
for line in open(fpInputTrainedGloveVector):
    lstContent.append(line)

print('length of pretrain {}'.format(len(lstContent)))
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

for file in arrFiles:
    if not file.endswith('csv'):
        continue
    fileCsv = fopDataset + file
    fpVectorItemCate=fopVectorAllSystems+file.replace('.csv','')+'_category.csv'
    fpVectorItemReg = fopVectorAllSystems + file.replace('.csv','') + '_regression.csv'

    raw_data = pd.read_csv(fileCsv)
    raw_data_2 = pd.read_csv(fileCsv)
    columnRegStory=raw_data_2['storypoint']
    raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0  # small
    raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), 'storypoint'] = 1  # medium
    raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), 'storypoint'] = 2  # large
    raw_data.loc[raw_data.storypoint > 15, 'storypoint'] = 3  # very large
    columnCateStory = raw_data['storypoint']

    titles_and_descriptions = []
    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), str(raw_data['description'][i])])
        titles_and_descriptions.append(str(strContent))

    text_after_tokenize = []
    for lineStr in titles_and_descriptions:
        lineAppend = preprocess(lineStr)
        text_after_tokenize.append(lineAppend)
    arrTokens = word_tokenize(str(text_after_tokenize[0]))
    vector0 = document_vector_glove(str(text_after_tokenize[i]),dictWordVectors)
    lenVectorOfWord = len(vector0)


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
        if not has_vector_representation(dictWordVectors, str(text_after_tokenize[i])):
            continue
        # arrTokens = word_tokenize(str(text_after_tokenize[i]))
        vector=document_vector_glove(str(text_after_tokenize[i]),dictWordVectors)
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

