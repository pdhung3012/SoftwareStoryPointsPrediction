
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from os import listdir
import os
from os.path import isfile, join
from UtilFunctions import createDirIfNotExist

fpModelData='../../../resultsSEE/trainedModels/d2v_all.model'
fp_allDatasetFolder="../PretrainData/"
fopDataset='../dataset/'
fopVector='../results/D2vML/vector/'
fopTextPreprocess='../results/D2vML/textProcess/'

createDirIfNotExist(fopVector)
createDirIfNotExist(fopTextPreprocess)

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


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



listAllTextAfterTokenizes=[]

for filename in os.listdir(fopDataset):
    if not filename.endswith(".csv"):
         # print(os.path.join(directory, filename))
        continue
    nameProject=filename.replace('.csv','')
    url=os.path.join(fopDataset, filename)
    raw_data = pd.read_csv(url)

    issue_titles = raw_data['title']
    issue_descriptions = raw_data['description']
    columnStoryPoints = raw_data['storypoint']


    titles_and_descriptions =  []
    for i in range(0, len(raw_data['description'])):
        strContent = ' '.join([str(raw_data['title'][i]), str(raw_data['description'][i])])
        titles_and_descriptions.append(str(strContent))

    '''
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
    '''

    text_after_tokenize=[]
    for lineStr in titles_and_descriptions:
        #lineAppend=preprocess(lineStr)
        lineAppend =lineStr
        text_after_tokenize.append(lineAppend)
        listAllTextAfterTokenizes.append(lineAppend)
    print('first {}!'.format(nameProject))

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(listAllTextAfterTokenizes)]
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
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save(fpModelData)
print("Model Saved")


for filename in os.listdir(fopDataset):
    if not filename.endswith(".csv"):
         # print(os.path.join(directory, filename))
        continue
    nameProject=filename.replace('.csv','')
    fpVector = fopVector+ nameProject+"_regression.csv"
    fpText = fopTextPreprocess+ nameProject+'_textInfo.csv'

    url=os.path.join(fopDataset, filename)
    raw_data = pd.read_csv(url)

    issue_titles = raw_data['title']
    issue_descriptions = raw_data['description']
    columnStoryPoints = raw_data['storypoint']

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
    print('complete {}!'.format(nameProject))

