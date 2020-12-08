from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np
from flask import Flask, request, jsonify
import pickle

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

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(data['protein1'])
    print(data['protein2'])
    input_data = np.zeros((1,96))
    prediction = model.predict(input_data)
    
    dic = {}
    dic['probability'] = prediction.tolist()
    dic['iteractions'] = True
    return jsonify(dic)

if __name__ == '__main__':
    app.run(port=5000, debug=True)