
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

## vectorization by pretrain w2v
import numpy as np
import gensim
# Load word2vec model (trained on Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary = True)

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
    return np.mean(model[doc], axis=0)

fpVector="data/vectorW2vCategories_pretrain_desc.csv"
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
    lstWord=word_tokenize(text_after_tokenize[i])
    if( not has_vector_representation(model,lstWord)):
        continue
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

