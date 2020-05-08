
import pandas as pd

fpModelData='data/d2v_all.model'
fpVector="data/vectorD2vCategories_all_desc.csv"
fpText="data/textD2vCategories_all_desc.txt"
fp_allDatasetFolder="../PretrainData/"



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
# titles_and_descriptions =  raw_data['title']+' '+raw_data['description']

titles_and_descriptions =  []
for i in range(0, len(raw_data['description'])):
    strContent = ' '.join([str(raw_data['title'][i]), str(raw_data['description'][i])])
    titles_and_descriptions.append(str(strContent))

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

def getTrainingTextForAllData(folderCsv,fpTextOutput):
    from os import listdir
    from os.path import isfile, join
    arrFiles = [f for f in listdir(folderCsv) if isfile(join(folderCsv, f))]
    fFile=open(fpTextOutput,'w')
    fFile.write('')
    fFile.close()
    index=0
    from nltk.corpus import stopwords
    item_stop_words = set(stopwords.words('english'))
    specific_stop_words = ['http', 'mule', 'studio']

    for file in arrFiles:
        listAllTextAfterTokenizes = []
        if not file.endswith('csv'):
            continue
        fileCsv=folderCsv+file
        item_data = pd.read_csv(fileCsv)
        # item_desc_org=item_data['title']+' '+item_data['description']
        item_desc=[]
        for i in range(0,len(item_data['description'])):
            strContent=' '.join([str(item_data['title'][i]),str(item_data['description'][i])])
            item_desc.append(str(strContent))

        # item_big_string = ' '.join(item_desc)
        # # Tokenize the string into words
        # item_tokens = word_tokenize(item_big_string)
        # # Remove non-alphabetic tokens, such as punctuation
        # item_words = [item_word.lower() for item_word in item_tokens if item_word.isalpha()]
        # Filter out stopwords

        # item_words = [w for w in item_words if w not in item_stop_words]
        # # Remove project specific stop words
        #
        # item_words = [w for w in item_words if w not in specific_stop_words]

        for it in item_desc:
            strIt=preprocessWithWordOnly(it)
            # strItem2=' '.join(strIt)
            # print(strIt+'\nhello world')
            listAllTextAfterTokenizes.append(strIt)
        print('{} {} {}'.format(file,len(item_desc), listAllTextAfterTokenizes[0]))
        # break
        fFile = open(fpTextOutput, 'a')
        strTotalItem='\n'.join(listAllTextAfterTokenizes)
        fFile.write(strTotalItem+'\n')
        fFile.close()

    # return listAllTextAfterTokenizes



def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word in words]
    doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)

def preprocessWithWordOnly(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in item_words]
    # doc = [word for word in doc if word.isalpha()]
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
# titles_and_descriptions

##create full text
# getTrainingTextForAllData(fp_allDatasetFolder,fpText)
# print(str(listAllTextAfterTokenizes))

fileText=open(fpText,'r')
listAllTextAfterTokenizes=fileText.read().split('\n')
fileText.close()

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(listAllTextAfterTokenizes)]

max_epochs = 5
vec_size = 20
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
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save(fpModelData)
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load(fpModelData)


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

