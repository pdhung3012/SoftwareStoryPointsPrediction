# Regression Example With Boston Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

import os

from os import listdir
from os.path import isfile, join

fopResult='result/keras_lstm_regression/'
fopDataset='../dataset/'

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

def convertFromTextToIndex(listStr,dictVocab):
	lstIndex=[]
	for item in listStr:
		arrTokens=word_tokenize(item)
		listItemIndex=[]
		for word in arrTokens:
			if not word in dictVocab:
				newIndex=len(dictVocab)+1
				dictVocab[word]=newIndex
				listItemIndex.append(str(newIndex))
			else:
				oldIndex=dictVocab[word]
				listItemIndex.append(str(oldIndex))
		# strIndex=' '.join(listItemIndex)
		lstIndex.append(listItemIndex)
	return lstIndex



# # load dataset
# dataframe = read_csv("data/housing.csv", delim_whitespace=True, header=None)
# dataset = dataframe.values
# # split into input (X) and output (Y) variables
# X = dataset[:,0:13]
# Y = dataset[:,13]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset

arrFiles = [f for f in listdir(fopDataset) if isfile(join(fopDataset, f))]
createDirIfNotExist(fopResult)
embedding_vecor_length = 100
top_words = 5000
fpMAEMin=fopResult+'MAEMin.txt'

o3 = open(fpMAEMin, 'w')
o3.write('')
o3.close()
fopOutputDetail=fopResult+'details/'
createDirIfNotExist(fopOutputDetail)
for file in arrFiles:
	if not file.endswith('csv'):
		continue
	fileCsv = fopDataset + file
	nameSystem=file.replace('.csv','')
	# fpVectorItemCate=fopResult+nameSystem+'_indexes_category.csv'

	fpOutputIndexReg = fopOutputDetail + nameSystem + '_indexes_regression.csv'
	fpOutputVocabReg = fopOutputDetail + nameSystem + '_vocab_regression.csv'
	fpOutputResultSum=fopOutputDetail+nameSystem+'_sum.txt'
	fpOutputResultPredict = fopOutputDetail + nameSystem + '_details.txt'

	raw_data = pd.read_csv(fileCsv)
	raw_data_2 = pd.read_csv(fileCsv)
	columnRegStory = raw_data_2['storypoint']
	# raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0  # small
	# raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 8), 'storypoint'] = 1  # medium
	# raw_data.loc[(raw_data.storypoint > 8) & (raw_data.storypoint <= 15), 'storypoint'] = 2  # large
	# raw_data.loc[raw_data.storypoint > 15, 'storypoint'] = 3  # very large
	# columnCateStory = raw_data['storypoint']

	titles_and_descriptions = []
	for i in range(0, len(raw_data['description'])):
		strContent = ' '.join([str(raw_data['title'][i]), str(raw_data['description'][i])])
		titles_and_descriptions.append(str(strContent))

	text_after_tokenize = []
	for lineStr in titles_and_descriptions:
		lineAppend = preprocess(lineStr)
		text_after_tokenize.append(lineAppend)
	list_str_index=[]
	dict_index={}
	list_str_index=convertFromTextToIndex(text_after_tokenize,dict_index)
	print('begin {}\t{}\t{}\t{}'.format(nameSystem,len(text_after_tokenize),len(list_str_index),len(columnRegStory)))
	X_train, X_test, y_train, y_test = train_test_split(list_str_index, columnRegStory, test_size=0.2, shuffle=False,
														stratify=None)

	max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(LSTM(100))
	model.add(Dense(80, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_absolute_error', optimizer='adam')

	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=128)
	# Final evaluation of the model
	predicted = model.predict(X_test)
	# predicted=model.predict(X_test, y_test)
	# print(predicted)
	np.savetxt(fpOutputResultPredict, predicted, fmt='%s', delimiter=',')
	maeAccuracy = mean_absolute_error(y_test, predicted)
	mqeAccuracy = mean_squared_error(y_test, predicted)
	o2 = open(fpOutputResultSum, 'w')
	# o2.write('Result for ' + str(classifier) + '\n')

	o2.write('MAE {}\nMQE {}\n'.format(maeAccuracy,mqeAccuracy))

	# o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
	# o2.write(str(confusion_matrix(all_label, predicted)) + '\n')
	# o2.write(str(classification_report(all_label, predicted)) + '\n')
	o2.close()

	o3 = open(fpMAEMin, 'a')
	o3.write('{}\t{}\n'.format(nameSystem,  maeAccuracy))
	o3.close()

	# estimators = []
	# estimators.append(('standardize', StandardScaler()))
	# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
	# pipeline = Pipeline(estimators)
	# kfold = KFold(n_splits=10)
	# results = cross_val_score(pipeline, X, Y, cv=kfold)
	# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))