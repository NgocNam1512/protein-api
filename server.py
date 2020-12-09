from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np
from flask import Flask, request, jsonify
# import pickle
import pandas as pd

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

app = Flask(__name__)
model = create_model()

emb = pd.read_csv("emb.csv")
emb = emb.drop('Unnamed: 0', axis=1)

proteinList = []
with open('proteinList.txt') as f:
    for line in f.readlines():
        proteinList.append(line.split("\t")[1].split(".")[0])

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)

    pr1 = data['protein1'].split(".")[0]
    index1 = proteinList.index(pr1)
    emb1 = emb.loc()

    pr2 = data['protein2'].split(".")[0]
    index2 = proteinList.index(pr2)

    input_data = np.hstack([emb.iloc[index1].to_numpy(), emb.iloc[index1].to_numpy()]).reshape((1,96))
    prediction = model.predict(input_data)
    
    dic = {}
    dic['probability'] = prediction.tolist()
    dic['iteractions'] = True
    return jsonify(dic)

if __name__ == '__main__':
    app.run(port=5000, debug=True)