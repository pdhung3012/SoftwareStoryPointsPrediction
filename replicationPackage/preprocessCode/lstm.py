
import pandas as pd
from nltk.tokenize import word_tokenize
import os
from flair.embeddings import WordEmbeddings, FlairEmbeddings, Sentence
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings



fopVectorAllSystems='../result/'
fopDataset='../data/datasetFromTSE2018/'






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

from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
glove_embedding = WordEmbeddings('glove')
document_lstm_embeddings = DocumentRNNEmbeddings([glove_embedding], rnn_type='LSTM')


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

    sentence = Sentence('The grass is green . And the sky is blue . I love you love love love lovedsdsds ddddd')

    document_lstm_embeddings.embed(sentence)
    vector0 =sentence.get_embedding()
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
        # if not has_vector(str(text_after_tokenize[i]),dictWordVectors,lenVectorOfWord):
        #     continue
        # arrTokens = word_tokenize(str(text_after_tokenize[i]))

        sentence = Sentence(str(text_after_tokenize[i]))

        document_lstm_embeddings.embed(sentence)

        vectorOrg=sentence.get_embedding()
        vector=[]
        for it2 in vectorOrg:
            vector.append(it2.data.item())
            # print(str(vectorOrg[0].data.item()))

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

