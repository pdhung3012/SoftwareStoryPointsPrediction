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
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error

nameSystem='talendesb'
fopInput='data/pretrainedVector/W2v/'
fopOutput='result/'


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.mkdir(fopOutput)
        print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

createDirIfNotExist(fopOutput)



nameClassifier='SVC'
fpVectorItemCate = fopInput+nameSystem  + '_regression.csv'
fpResultAll=fopOutput+'summary.txt'
fpDetailsDefaultClassifier=fopOutput+'details_default.txt'
fpDetailsTunedClassifier=fopOutput+'details_tune.txt'

df_all = pd.read_csv(fpVectorItemCate)
print(list(df_all.columns.values))
all_label = df_all['story']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','story'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size = 0.2,shuffle = False, stratify = None)
kfold=StratifiedKFold(10,shuffle=True)

classifier=xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
classifier.fit(X_train,y_train)
class_predictions =classifier.predict(X_test)
class_maeAccuracy = mean_absolute_error(y_test, class_predictions)
class_mqeAccuracy = mean_squared_error(y_test, class_predictions)
# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }
# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
parameters = {'objective':['reg:linear'],
            'colsample_bytree' : [0.3],
              'learning_rate': [0.1], #so called `eta` value
              'max_depth': [ 1,3,5],
              'alpha':[10,20],

              # 'min_child_weight': [4],
              # 'silent': [1],
              # 'subsample': [0.7],
              # 'colsample_bytree': [0.7],
              'n_estimators': [10]}

grid = GridSearchCV(classifier,parameters,scoring='neg_mean_absolute_error',refit=True,verbose=2)
grid.fit(X_train,y_train)
# grid_predictions = grid.predict(X_test)
# grid_maeAccuracy = mean_absolute_error(y_test, grid_predictions)
# grid_mqeAccuracy = mean_squared_error(y_test, grid_predictions)

best_class=grid.best_estimator_
grid_predictions = best_class.predict(X_test)
grid_maeAccuracy = mean_absolute_error(y_test, grid_predictions)
grid_mqeAccuracy = mean_squared_error(y_test, grid_predictions)

# classifier.fit(X_train,y_train)
# grid_predictions = classifier.predict(X_test)
# cross_val = cross_val_score(grid.best_estimator_, all_data, all_label, cv=kfold, n_jobs=1)
# grid_predictions =cross_val_predict(grid.best_estimator_, all_data, all_label, cv=kfold)
o2 = open(fpResultAll, 'w')
o2.write('Default estimator:\n'+str(classifier)+'\n')
o2.write('MAE: {}\nMQE: {}\n'.format(class_maeAccuracy,class_mqeAccuracy))
# o2.write(str(confusion_matrix(all_label,class_predictions))+'\n')
# o2.write(str(classification_report(all_label,class_predictions))+'\n')
# o2.write(str(accuracy_score(all_label,class_predictions))+'\n\n\n')

# print(str(grid.estimator))

o2.write('Best estimator:\n'+str(grid.estimator)+'\n')
o2.write('Result for best estimator of ' + nameClassifier + '\n')
# o2.write(str(confusion_matrix(y_test,grid_predictions))+'\n')
# o2.write(str(classification_report(y_test,grid_predictions))+'\n')
o2.write('MAE: {}\nMQE: {}\n'.format(grid_maeAccuracy,grid_mqeAccuracy))

o2.close()
np.savetxt(fpDetailsDefaultClassifier, class_predictions, fmt='%s', delimiter=',')
np.savetxt(fpDetailsTunedClassifier, grid_predictions, fmt='%s', delimiter=',')
