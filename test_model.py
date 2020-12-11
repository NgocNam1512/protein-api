from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np
from flask import Flask, request, jsonify
# import pickle
import pandas as pd
import json
from flask_cors import CORS

def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=96))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))

    model.load_weights("model_weights.h5")
    print("Loaded model from disk")

    return model

model = create_model()
emb = pd.read_csv("emb.csv")
emb = emb.drop('Unnamed: 0', axis=1)
print(emb.shape)

proteinList = []
with open('proteinList.txt') as f:
    for line in f.readlines():
        proteinList.append(line.split("\t")[1].split(".")[0])

testList = []
with open('NegativeEdges.txt') as f:
    for line in f.readlines():
        pair = [int(line.split("\t")[0]), int(line.split("\t")[1].replace("\n", ""))]
        testList.append(pair)

finalList = []
X = np.empty((len(testList),2*emb.shape[1]))
k = 0
for x in testList:
    try:
        X[k] = np.hstack((emb.iloc[x[0]].to_numpy(),emb.iloc[x[1]].to_numpy()))
        finalList.append(x)
        k = k + 1 
    except:
        continue
print(X)
Y = np.full((k,2),[1,0])
y_prob = model.predict(X)
# y_classes = y_prob.argmax(axis=-1)
for i, prob in enumerate(y_prob):
    if np.argmax(prob) == 0:
        print(finalList[i])
        