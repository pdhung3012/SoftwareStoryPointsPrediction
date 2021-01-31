"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a simple GNN in disjoint mode.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from spektral.data import DisjointLoader
from spektral.datasets import QM9
from spektral.layers import ECCConv, GlobalSumPool
from GenerateListOfGraphForStoryPoints import extractGraphSpekFromText,preprocess,initDefaultTextEnvi

################################################################################
# LOAD STORY POINTS
################################################################################

fopDataset = '../dataset/'
fnSystem='mulestudio.csv'
fileCsv = fopDataset + fnSystem

raw_data = pd.read_csv(fileCsv)
raw_data_2 = pd.read_csv(fileCsv)
columnId = raw_data['issuekey']
columnRegStory = raw_data_2['storypoint']
titles_and_descriptions = []
for i in range(0, len(raw_data['description'])):
    strContent = ' '.join([str(raw_data['title'][i]), ' . ', str(raw_data['description'][i])])
    strContent=preprocess(strContent)
    titles_and_descriptions.append(str(strContent))

nlp_model,nlp=initDefaultTextEnvi()
dictVocab={}

listGraphs=[]

for i in range(0,len(titles_and_descriptions)):
    itemLabel=columnRegStory[i]
    graphSpek=extractGraphSpekFromText(titles_and_descriptions[i],itemLabel,dictVocab,nlp_model,nlp)
    listGraphs.append(graphSpek)
#dataset=np.asarray(listGraphs)


################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

################################################################################
# LOAD DATA
################################################################################
#dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

print('size see {}'.format(len(listGraphs)))
nLenSEE=len(listGraphs)
dataset = QM9(amount=nLenSEE)  # Set amount=None to train on whole dataset
oldds=dataset

print('type {}'.format(type(dataset)))
for i in range(0,nLenSEE):
   # print('aaa {} {}'.format(i, dataset[i].a))
    dataset[i].a=listGraphs[i].a
    #print('bbb {} {}'.format(i,dataset[i].a))
    if i==0:
        break

#    dataset[i].y=oldds[i].y


print('info \n{}\n{}\n'.format(dataset[0].y,dataset[1].y))
# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target
print('nout {} {} {}'.format(F,S,n_out))

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
#print('idx {}'.format(idx_tr))
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True, name="A_in")
E_in = Input(shape=(S,), name="E_in")
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = MeanSquaredError()


################################################################################
# FIT MODEL
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print("Fitting model")
current_batch = 0
model_loss = 0
for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        print("Loss: {}".format(model_loss / loader_tr.steps_per_epoch))
        model_loss = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model_loss = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    model_loss += loss_fn(target, predictions)
model_loss /= loader_te.steps_per_epoch
print("Done. Test loss: {}".format(model_loss))
