
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import numpy as np
import gensim

fopVectorAllSystems='../result/'
fopDataset='../data/datasetFromTSE2018/'
print('Remember to download the google pretrained model GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/ and extract it inside data folder')

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


# Load word2vec model (trained on Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary = True)
# # Check dimension of word vectors
# model.vector_size
# # Filter the list of vectors to include only those that Word2Vec has a vector for
# vector_list = [model[word] for word in words if word in model.vocab]
# # Create a list of the words corresponding to these vectors
# words_filtered = [word for word in words if word in model.vocab]
# # Zip the words together with their vector representations
# word_vec_zip = zip(words_filtered, vector_list)
#
# # Cast to a dict so we can turn it into a DataFrame
# word_vec_dict = dict(word_vec_zip)
# df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
# df.head(3)
# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    # print("{} docs removed".format(number_of_docs - len(corpus)))
    return (corpus, texts)




# Averaging Word Embeddings
def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.vocab]
    print(doc)
    return np.mean(model[doc], axis=0)


from os import listdir
from os.path import isfile, join
arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopVectorAllSystems)

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
    vector0 = document_vector(model, arrTokens)
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
        arrTokens = word_tokenize(str(text_after_tokenize[i]))
        if not has_vector_representation(model ,arrTokens):
            continue
        vector = document_vector(model, arrTokens)
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

