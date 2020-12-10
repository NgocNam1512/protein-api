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

app = Flask(__name__)
model = create_model()

emb = pd.read_csv("emb.csv")
emb = emb.drop('Unnamed: 0', axis=1)
CORS(app)
proteinList = []
with open('proteinList.txt') as f:
    for line in f.readlines():
        proteinList.append(line.split("\t")[1].split(".")[0])

proteinName = {}
with open('result.json') as json_file:
    proteinName = json.load(json_file)

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    found = True

    pr1 = data['protein1'].split(".")[0]
    index1 = proteinList.index(pr1)

    pr2 = data['protein2'].split(".")[0]
    index2 = proteinList.index(pr2)

    for pr in proteinName:
        if pr['code'] == pr1:
            pr1_name = pr['name']
        if pr['code'] == pr2:
            pr2_name = pr['name']

    input_data = np.hstack([emb.iloc[index1].to_numpy(), emb.iloc[index2].to_numpy()]).reshape((1,96))
    prediction = model.predict(input_data)
    
    dic = {}
    dic['name'] = {"protein1":pr1_name, "protein2":pr2_name}
    dic['probability'] = prediction.tolist()
    dic['iteractions'] = True
    return jsonify(dic)

if __name__ == '__main__':
    app.run(port=5000, debug=True)