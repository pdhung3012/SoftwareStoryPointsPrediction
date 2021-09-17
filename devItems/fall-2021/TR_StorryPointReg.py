import glob
import sys, os
import operator

from tree_sitter import Language, Parser
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../../')))
from UtilFunctions import createDirIfNotExist
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from langdetect import detect
from sklearn.metrics import confusion_matrix
import langid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error

fopRootData='../../../dataPapers/SEE/'
createDirIfNotExist(fopRootData)
fopResultSystems=fopRootData+'result_tfidf_dmOnly/'
fileCsv='../dataset/datamanagement.csv'
createDirIfNotExist(fopResultSystems)

raw_data = pd.read_csv(fileCsv)

columnId = raw_data['issuekey']
columnRegStory = raw_data['storypoint']

titles = []
descs=[]
lstTups=[]
for i in range(0, len(raw_data['description'])):
    tit=str(raw_data['title'][i])
    desc=str(raw_data['description'][i])
    score=int(raw_data['storypoint'][i])
    id=str(raw_data['issuekey'][i])
    mlScore=-1
    distance=-1
    tup=(i,id,distance,score,mlScore,tit,desc)
    lstTups.append(tup)
    titles.append(tit)
    descs.append(desc)


tup_train, tup_test, y_train, y_test = train_test_split(lstTups, columnRegStory, test_size = 0.2,shuffle = False, stratify = None)

vectorizer = TfidfVectorizer(ngram_range=(1, 4),max_features=1000)
model = vectorizer.fit(titles)
X_Train=[x[5] for x in tup_train]
X_Test=[x[5] for x in tup_test]
# vec_total_all=model.transform(lstAllText).toarray()
vec_train_tit=model.transform(X_Train).toarray()
vec_test_tit=model.transform(X_Test).toarray()


vectorizer = TfidfVectorizer(ngram_range=(1, 4),max_features=1000)
model = vectorizer.fit(descs)
X_Train=[x[6] for x in tup_train]
X_Test=[x[6] for x in tup_test]
# vec_total_all=model.transform(lstAllText).toarray()
vec_train_desc=model.transform(X_Train).toarray()
vec_test_desc=model.transform(X_Test).toarray()


#pca = PCA(n_components=100)
print('prepare to fit transform')
# vec_train=vec_train_all
# vec_testP=vec_testP_all
# vec_testW=vec_testW_all
import numpy as np
vec_train= np.concatenate([vec_train_tit,vec_train_tit],axis=1)
vec_test=np.concatenate([vec_test_desc,vec_test_desc],axis=1)

print('end fit transform')

rf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1)

print('go here')
start = time.time()
rf_model = rf.fit(vec_train, y_train)
# filename4 = fop+arrConfigs[idx]+ '_mlmodel.bin'
end = time.time()
fit_time = (end - start)
print('end train {}'.format(fit_time))
start = time.time()
y_pred = rf_model.predict(vec_test)
end = time.time()
pred_time = (end - start)

lstTups=[]
for i in range(0,len(y_pred)):
    tupTestIt=tup_test[i]
    expectedScore=tupTestIt[4]
    distance=expectedScore-y_pred[i]
    lst = list(tupTestIt)
    lst[4] = y_pred[i]
    lst[2]=distance
    tup_test[i]=tup(lst)
lstTups.sort(key = lambda x: x[2],reverse=True)

lstStrTup=['i,id,distance,score,mlScore,tit,desc']
for tup in lstTups:
    strItem=','.join(list(tup))
    lstStrTup.append(strItem)
fpOutResultTest=fopResultSystems+'resultTest.csv'
f2=open(fpOutResultTest,'w')
f2.write('\n'.join(lstStrTup))
f2.close()
print('end test {}'.format(pred_time))


fpOutResultDetail=fopResultSystems+'summary.txt'
f1=open(fpOutResultDetail,'w')
maeAccuracy = mean_absolute_error(y_test, y_pred)
mqeAccuracy = mean_squared_error(y_test, y_pred)
f1.print('mae: {}\nmqe: {}\n'.format(maeAccuracy,mqeAccuracy))
f1.close()

