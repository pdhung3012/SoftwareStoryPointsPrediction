
import pandas as pd




url = 'https://raw.githubusercontent.com/SEAnalytics/datasets/master/storypoint/IEEE%20TSE2018/dataset/mulestudio.csv'
raw_data = pd.read_csv(url)
raw_data.columns
raw_data.head(6)

raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0 #small
raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), 'storypoint'] = 1 #medium
raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), 'storypoint'] = 2 #large
raw_data.loc[raw_data.storypoint > 15, 'storypoint'] = 3 #very large

issue_titles = raw_data['title']
issue_descriptions = raw_data['description']
columnStoryPoints = raw_data['storypoint']

# cocat title and description
titles_and_descriptions = raw_data['title'] +' ' + raw_data['description']

big_string = ' '.join(titles_and_descriptions)

from nltk.tokenize import word_tokenize

# Tokenize the string into words
tokens = word_tokenize(big_string)

# Remove non-alphabetic tokens, such as punctuation
words = [word.lower() for word in tokens if word.isalpha()]

# Filter out stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if w not in stop_words]

# Remove project specific stop words
specific_stop_words = ['http', 'mule', 'studio']
words = [w for w in words if w not in specific_stop_words]

def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word in words]
    doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

text_after_tokenize=[]
for lineStr in titles_and_descriptions:
    lineAppend=preprocess(lineStr)
    text_after_tokenize.append(lineAppend)

print(text_after_tokenize)
big_processed_string='\n'.join(text_after_tokenize)

fTextForTrain=open('data/textProcessGlove.txt','w')
fTextForTrain.write(big_processed_string)
fTextForTrain.close()

## vectorization
import numpy as np

def categorize(score):
    result=0
    if score <=2:
        result=0
    elif score>2 and score<=8:
        result=1
    elif score>8 and score<=15:
        result=2
    else:
        result=3
    # return score
    return result

# Averaging Word Embeddings
def document_vector(strContent,dictWordVectors,lenVector):
    # remove out-of-vocabulary words
    arrWord=strContent.split(' ')
    listVectorWords=[]
    for word in arrWord:
        if word in dictWordVectors:
            arr=dictWordVectors[word]
            listVectorWords.append(arr)

    if len(listVectorWords)==0:
        arrResult=[]
        for i in range(0,lenVector):
            arrResult.append(0)
        return arrResult
    return np.mean(listVectorWords, axis=0)

fpVector="data/vectorGloveCastegories.csv"
fpInputTrainedGloveVector="data/glove_words_v2.txt"

dictWordVectors={}
fRead=open(fpInputTrainedGloveVector,'r')
strVectorContent=fRead.read()
fRead.close()
lstContent=strVectorContent.split('\n')
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

columnTitleRow = "no,story,"
for i in range(0,lenVectorOfWord):
    item='feature-'+str(i+1)
    columnTitleRow = ''.join([columnTitleRow, item])
    if i!=lenVectorOfWord-1:
        columnTitleRow = ''.join([columnTitleRow,  ","])
columnTitleRow = ''.join([columnTitleRow, "\n"])
csv = open(fpVector, 'w')
csv.write(columnTitleRow)

corpusVector = []
for i in range(0,len(text_after_tokenize)):
    vector=document_vector(str(text_after_tokenize[i]),dictWordVectors,lenVectorOfWord)
    corpusVector.append(vector)
    # strVector=','.join(vector)
    strCate=str(columnStoryPoints[i])
    # strRow=''.join([str(i+1),',','S-'+str(columnStoryPoints[i]),])
    # strRow = ''.join([str(i + 1), ',', 'S-' + strCate, ])
    strRow = ''.join([str(i + 1), ',', '' + strCate, ])
    for j in range(0,lenVectorOfWord):
        strRow=''.join([strRow,',',str(vector[j])])
    strRow = ''.join([strRow, '\n'])
    csv.write(strRow)
csv.close()

