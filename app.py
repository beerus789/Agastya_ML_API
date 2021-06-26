from flask import Flask, json, jsonify, request
from model_files.model import predict_covid

app = Flask(__name__)

@app.route('/')
def index():
    return 'Ping Successful - This is the official API for Agastya service.'

@app.route('/predict', methods = ['POST'])
def predict():
    image_loc = request.data
    # print(image_loc)

    # return req
    res = predict_covid(image_loc)
    # print(res)
    response = {
        "type": res[0],
        "accuracy": res[1]
    }
    return jsonify(response)
