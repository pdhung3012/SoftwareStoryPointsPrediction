
import pandas as pd

fpModelData='data/d2v.model'
fpVector="data/vectorD2vCategories_desc.csv"



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
titles_and_descriptions =  raw_data['description']

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

# print(text_after_tokenize)
# big_processed_string='\n'.join(text_after_tokenize)
#
# fTextForTrain=open('data/textProcessGlove.txt','w')
# fTextForTrain.write(big_processed_string)
# fTextForTrain.close()

## vectorization by d2v

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# data = ["I love machine learning. Its awesome.",
#         "I love coding in python",
#         "I love building chatbots",
#         "they chat amagingly well"]
titles_and_descriptions

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(titles_and_descriptions)]

max_epochs = 500
vec_size = 50
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save(fpModelData)
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load(fpModelData)

# fRead=open(fpInputTrainedGloveVector,'r')
# strVectorContent=fRead.read()
# fRead.close()
# lstContent=[]
# for line in open(fpInputTrainedGloveVector):
#     lstContent.append(line)
#
# print('length of pretrain {}'.format(len(lstContent)))
# lenVectorOfWord=0
# for i in range(0,len(lstContent)):
#     lstVectorItem=[]
#     word=''
#     arrItem=lstContent[i].split(' ')
#     # print(str(arrItem))
#     if len(arrItem)>2:
#         word=arrItem[0]
#         if lenVectorOfWord==0:
#             lenVectorOfWord=len(arrItem)-1
#         for j in range(1,len(arrItem)):
#             lstVectorItem.append(float(arrItem[j]) )
#         dictWordVectors[word]=lstVectorItem

arrTokens=word_tokenize(str(text_after_tokenize[0]))
vector0=model.infer_vector(arrTokens)
lenVectorOfWord=len(vector0)

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
    # if not has_vector(str(text_after_tokenize[i]),dictWordVectors,lenVectorOfWord):
    #     continue
    arrTokens = word_tokenize(str(text_after_tokenize[0]))
    vector=model.infer_vector(arrTokens)
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

