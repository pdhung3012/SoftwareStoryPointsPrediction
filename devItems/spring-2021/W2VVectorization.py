import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from UtilFunctions import createDirIfNotExist

import nltk
nltk.download('stopwords')

fopDataset='../dataset/'
fopW2V='../../../resultsSEE/trainedModels/GoogleNews-vectors-negative300.bin'
fopVector='../results/W2vML/vector/'

createDirIfNotExist(fopVector)
import numpy as np
import gensim


# Load word2vec model (trained on Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format(fopW2V, binary=True)

arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
for filename in os.listdir(fopDataset):
    if not filename.endswith(".csv"):
         # print(os.path.join(directory, filename))
        continue
    nameProject=filename.replace('.csv','')
    url=os.path.join(fopDataset, filename)
    raw_data = pd.read_csv(url)
    raw_data.columns
    raw_data.head(6)
    '''
    columnCatName='sp_cat'
    raw_data.loc[raw_data.storypoint <= 2, columnCatName] = 0 #small
    raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), columnCatName] = 1 #medium
    raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), columnCatName] = 2 #large
    raw_data.loc[raw_data.storypoint > 15, columnCatName] = 3 #very large
    '''
    issue_titles = raw_data['title']
    issue_descriptions = raw_data['description']
    columnStoryPoints = raw_data['storypoint']

    # cocat title and description
    titles_and_descriptions =[]
    for i in range(0,len(raw_data)):
        strContent=str(raw_data['title'][i])+' '+str(raw_data['description'][i])
        #raw_data['title'] +' ' + raw_data['description']
        titles_and_descriptions.append(strContent)

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
    specific_stop_words = []
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

    ## vectorization by pretrain w2v
    i

    # Check dimension of word vectors
    model.vector_size

    # Filter the list of vectors to include only those that Word2Vec has a vector for
    vector_list = [model[word] for word in words if word in model.vocab]

    # Create a list of the words corresponding to these vectors
    words_filtered = [word for word in words if word in model.vocab]

    # Zip the words together with their vector representations
    word_vec_zip = zip(words_filtered, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    df.head(3)


    # Averaging Word Embeddings
    # Averaging Word Embeddings
    def document_vector(word2vec_model, doc):
        # remove out-of-vocabulary words
        doc = [word for word in doc if word in model.vocab]
        return np.mean(model[doc], axis=0)

    fpVector=fopVector+nameProject+'_regression.csv'
    # fpInputTrainedGloveVector="data/glovePretrain/glove.840B.300d.txt"

    dictWordVectors={}
    vector0=document_vector(model,str(text_after_tokenize[0]))
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
        vector=document_vector(model,str(text_after_tokenize[i]))
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

