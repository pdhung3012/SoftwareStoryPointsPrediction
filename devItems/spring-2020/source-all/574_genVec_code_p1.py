from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim
from sklearn.decomposition import PCA

import codecs
fopVectorAllSystems='data/vector574_Tfidf1_code/'
fopDataset='dataset574/'

def categorize_178_328_741(score):
    result=0
    if score <=178:
        result=0
    elif score>178 and score<=328:
        result=1
    elif score>328 and score<=741:
        result=2
    else:
        result=3
    # return score
    return result

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




from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)


fileCsvTextTrain = fopDataset + 'text_train.txt'
fileCsvTextTest = fopDataset + 'text_test.txt'
fileCsvStarTrain = fopDataset + 'star_train.csv'
fileCsvStarTest = fopDataset + 'star_test.csv'
fpVectorItemTrainCate=fopVectorAllSystems+'code_train_category.csv'
fpVectorItemTrainReg = fopVectorAllSystems + 'code_train_regression.csv'
fpVectorItemTestCate=fopVectorAllSystems+'code_test_category.csv'
fpVectorItemTestReg = fopVectorAllSystems + 'code_test_regression.csv'

# raw_data = pd.read_csv(fileCsv)
# raw_data_2 = pd.read_csv(fileCsv)
# columnRegStory=raw_data_2['storypoint']
# raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0  # small
# raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), 'storypoint'] = 1  # medium
# raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), 'storypoint'] = 2  # large
# raw_data.loc[raw_data.storypoint > 15, 'storypoint'] = 3  # very large
# columnCateStory = raw_data['storypoint']

columnCodeTrain = []
columnASTTrain = []
columnStarCateTrain = []
columnStarRegTrain = []
columnCodeTest = []
columnASTTest = []
columnStarCateTest = []
columnStarRegTest = []
fCode=codecs.open(fileCsvTextTrain, 'r', encoding='utf-8',errors='ignore')
for line in fCode:
    fields=line.split(',')
    if len(fields)>=3:
        columnCodeTrain.append(fields[2])
    else:
        columnCodeTrain.append('')
fCode.close()

fCode=codecs.open(fileCsvTextTest, 'r', encoding='utf-8',errors='ignore')
for line in fCode:
    fields=line.split(',')
    if len(fields)>=3:
        columnCodeTest.append(fields[2])
    else:
        columnCodeTest.append('')

fCode.close()

fCode=open(fileCsvStarTrain, 'r')
for line in fCode:
    fields=line.split(',')
    columnStarCateTrain.append(categorize_178_328_741(int (fields[2])))
    columnStarRegTrain.append(int(fields[2]))
fCode.close()

fCode=open(fileCsvStarTest, 'r')
for line in fCode:
    fields=line.split(',')
    columnStarCateTest.append(categorize_178_328_741(int(fields[2])))
    columnStarRegTest.append(int(fields[2]))
fCode.close()

text_after_tokenize = []
for lineStr in columnCodeTrain:
    text_after_tokenize.append(lineStr)
for lineStr in columnCodeTest:
    text_after_tokenize.append(lineStr)
# get vector using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(text_after_tokenize)
X = X.toarray()
print('before vectorize')
# X = PCA().fit(X)
pca = PCA(n_components=50)
X = pca.fit_transform(X)
print('end vectorize')

lenVectorOfWord = len(X[0])


columnTitleRow = "no,star,"
for i in range(0,lenVectorOfWord):
    item='feature-'+str(i+1)
    columnTitleRow = ''.join([columnTitleRow, item])
    if i!=lenVectorOfWord-1:
        columnTitleRow = ''.join([columnTitleRow,  ","])
columnTitleRow = ''.join([columnTitleRow, "\n"])

# write csv to test
csv = open(fpVectorItemTestCate, 'w')
csv.write(columnTitleRow)
csv2 = open(fpVectorItemTestReg, 'w')
csv2.write(columnTitleRow)
corpusVector = []
for i in range(len(columnCodeTrain),len(text_after_tokenize)):
    vector= X[i]
    corpusVector.append(vector)
    index = i - len(columnCodeTrain) + 1
    strCate=str(columnStarCateTest[index-1])
    strReg=str(columnStarRegTest[index-1])

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

# write train to csv
csv = open(fpVectorItemTrainCate, 'w')
csv.write(columnTitleRow)
csv2 = open(fpVectorItemTrainReg, 'w')
csv2.write(columnTitleRow)
corpusVector = []
for i in range(0,len(columnCodeTrain)):
    vector= X[i]
    corpusVector.append(vector)
    strCate=str(columnStarCateTrain[i])
    strReg=str(columnStarRegTrain[i])
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



print('Finish {}')

