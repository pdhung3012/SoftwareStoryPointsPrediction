import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
import os
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

nameSystem='code'
fopInput='data/vector574_Tfidf1/'
fopOutput='result/tune_574_tfidf1_'+nameSystem+'_cate/'
createDirIfNotExist(fopOutput)



nameClassifier='SVC'
fpVectorItemTrainCate = fopInput+nameSystem  + '_train_category.csv'
fpVectorItemTestCate = fopInput+nameSystem  + '_test_category.csv'
fpResultAll=fopOutput+'summary.txt'
fpDetailsDefaultClassifier=fopOutput+'details_default.txt'
fpDetailsTunedClassifier=fopOutput+'details_tune.txt'

df_train = pd.read_csv(fpVectorItemTrainCate)
print(list(df_train.columns.values))
y_train = df_train['star']
X_train = df_train.drop(['no','star'],axis=1)

df_test = pd.read_csv(fpVectorItemTestCate)
print(list(df_test.columns.values))
y_test = df_test['star']
X_test = df_test.drop(['no','star'],axis=1)

kfold=StratifiedKFold(10,shuffle=True)

classifier=SVC()
classifier.fit(X_train,y_train)
# class_predictions =cross_val_predict(classifier, all_data, all_label, cv=kfold)
class_predictions =classifier.predict(X_test)
# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }
# , 'gamma': [1,0.1,0.01,0.001]
# 'C': [0.1,1, 10, 100]
param_grid = {'C': [1],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(classifier,param_grid,cv=kfold,refit=True,verbose=2)
grid.fit(X_train,y_train)
# grid_predictions = grid.predict(X_test)

# classifier.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
# cross_val = cross_val_score(grid.best_estimator_, all_data, all_label, cv=kfold, n_jobs=1)
# grid_predictions =cross_val_predict(grid.best_estimator_, all_data, all_label, cv=kfold)
o2 = open(fpResultAll, 'w')
o2.write('Default estimator:\n'+str(classifier)+'\n')
o2.write(str(confusion_matrix(y_test,class_predictions))+'\n')
o2.write(str(classification_report(y_test,class_predictions))+'\n')
o2.write(str(accuracy_score(y_test,class_predictions))+'\n\n\n')

o2.write('Best estimator:\n'+str(grid.best_estimator_)+'\n')
o2.write('Result for best estimator of ' + nameClassifier + '\n')
# o2.write(str(confusion_matrix(y_test,grid_predictions))+'\n')
# o2.write(str(classification_report(y_test,grid_predictions))+'\n')
o2.write(str(confusion_matrix(y_test,grid_predictions))+'\n')
o2.write(str(classification_report(y_test,grid_predictions))+'\n')
o2.write(str(accuracy_score(y_test,grid_predictions))+'\n')

o2.close()
np.savetxt(fpDetailsDefaultClassifier, class_predictions, fmt='%s', delimiter=',')
np.savetxt(fpDetailsTunedClassifier, grid_predictions, fmt='%s', delimiter=',')
# try:
#     filePredict = ''.join([fopOutputItemDetail, file, '_', arrClassifierName[index - 1], '.txt'])
#     print("********", "\n", "10 fold CV Results with: ", str(classifier))
#     cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
#     predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
#     weightAvg = precision_score(all_label, predicted, average='weighted') * 100
#     normalAvg = accuracy_score(all_label, predicted) * 100
#     print('{:.2f}'.format(normalAvg))
#
#     np.savetxt(filePredict, predicted, fmt='%s', delimiter=',')
#     o2 = open(fpResultAll, 'a')
#     o2.write('Result for ' + str(classifier) + '\n')
#     o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
#     o2.write(str(confusion_matrix(all_label, predicted)) + '\n')
#     o2.write(str(classification_report(all_label, predicted)) + '\n')
#     o2.close()
#
#     strClassX = str(arrClassifierName[index - 1])
# except Exception as inst:
#     print("Error ")
#     print(type(inst))  # the exception instance
#     print(inst.args)  # arguments stored in .args
#     print(inst)
