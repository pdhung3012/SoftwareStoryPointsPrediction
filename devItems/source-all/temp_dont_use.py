import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

fopInput='data/testVectorSystems_Tfidf4/'
fopOutput='result/tune_tfidf4/'
createDirIfNotExist(fopOutput)


nameSystem='moodle'

nameClassifier='MLP'
fpVectorItemCate = fopInput+nameSystem  + '_category.csv'
fpResultAll=fopOutput+'summary.txt'
fpDetails=fopOutput+'details.txt'

df_all = pd.read_csv(fpVectorItemCate)
print(list(df_all.columns.values))
all_label = df_all['story']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','story'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2,shuffle = False, stratify = None)

classifier=MLPClassifier(max_iter=100)

kfold=StratifiedKFold(10)
parameter_space = {
    'hidden_layer_sizes': [ (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
grid = GridSearchCV(classifier,parameter_space,refit=False,cv=kfold)
grid.fit(all_data,all_label)

grid_predictions = grid.predict(all_data)
o2 = open(fpResultAll, 'w')
o2.write('Best estimator:\n'+str(grid.best_estimator_)+'\n')
o2.write('Result for best estimator of ' + nameClassifier + '\n')
o2.write(str(confusion_matrix(y_test,grid_predictions))+'\n')
o2.write(str(classification_report(y_test,grid_predictions))+'\n')
o2.close()
np.savetxt(fpDetails, grid_predictions, fmt='%s', delimiter=',')
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
