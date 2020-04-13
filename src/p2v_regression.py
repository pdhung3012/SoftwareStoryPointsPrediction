# -*- coding: utf-8 -*-
"""SoftwareEffortEstimation_Paragraph2Vec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10dMa09XK6GynsE4d5gRB2r8xZrEzquvO
"""


import pandas as pd

# Load paragraph to vector representation for software issue/request description
paragraph2vec = pd.read_csv('data/description_mulestudio_paragraph2vec.csv')

paragraph2vec.head(3)

url = 'https://raw.githubusercontent.com/SEAnalytics/datasets/master/storypoint/IEEE%20TSE2018/dataset/mulestudio.csv'
raw_data = pd.read_csv(url)
raw_data.columns

# raw_data.loc[raw_data.storypoint <= 2, 'storypoint'] = 0 #small
# raw_data.loc[(raw_data.storypoint > 2) & (raw_data.storypoint <= 5), 'storypoint'] = 1 #medium
# raw_data.loc[raw_data.storypoint > 5, 'storypoint'] = 2 #big

y = raw_data['storypoint']

# 80% of data goes to training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(paragraph2vec, y, test_size = 0.2, random_state = 10)

from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
# Instantiate an XGBRegressor
xgr = xgb.XGBRegressor(random_state=2)

# Fit the classifier to the training set
xgr.fit(X_train, y_train)

y_pred = xgr.predict(X_test)

mqeAccuracy=mean_squared_error(y_test, y_pred)
maeAccuracy=mean_absolute_error(y_test, y_pred)



# Accuracy
y_pred_rounded = [round(prediction,0) for prediction in y_pred ]
y_pred_rounded = [int(prediction) for prediction in y_pred_rounded]

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
acc_score = metrics.accuracy_score(y_test, y_pred_rounded)
print('Total accuracy classification score: {}'.format(acc_score))
print(str(confusion_matrix(y_test,y_pred_rounded)) + '\n')
print(str(classification_report( y_test,y_pred_rounded)) + '\n')
print('MAE: {} and MSE: {}'.format(maeAccuracy,mqeAccuracy))
# # Build the classifier
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
#
# # Fit the classifier to the training set
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# # Accuracy
# from sklearn import metrics
# acc_score = metrics.accuracy_score(y_test, y_pred)
# print('Total accuracy classification score: {}'.format(acc_score))
