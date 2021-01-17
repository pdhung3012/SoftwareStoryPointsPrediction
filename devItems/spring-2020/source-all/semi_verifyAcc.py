import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import os
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error

fpInputResult='dataVerifyAcc/adv-result.txt'
fpOutputResult='dataVerifyAcc/details-adv-result.txt'

fIn=open(fpInputResult,'r')
arrIn=fIn.read().split('\n')

predicted=[]
expected=[]
for item in arrIn:
    if len(item.split('\t'))<2:
        continue
    pItem=item.split('\t')[0].replace('tensor(','').replace(', device=\'cuda:0\')','')
    eItem=item.split('\t')[1]
    predicted.append(pItem)
    expected.append(eItem)

o2 = open(fpOutputResult, 'w')
o2.write('Detail \n')
# o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
o2.write(str(confusion_matrix(expected, predicted)) + '\n')
o2.write(str(classification_report(expected, predicted)) + '\n')
normalAvg = accuracy_score(expected, predicted) * 100
o2.write('Accuracy: {}'.format(normalAvg))
o2.close()