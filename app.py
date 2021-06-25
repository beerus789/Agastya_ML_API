from flask import Flask, json, jsonify, request
from model_files.model import predict_covid

app = Flask("covid_prediction")

@app.route('/')
def index():
    return 'Ping Successful'

@app.route('/predict', methods = ['POST'])
def predict():
    image_loc = request.data
    print(image_loc)

    # return req
    res = predict_covid(image_loc)
    print(res)
    response = {
        "type": res[0],
        "accuracy": res[1]
    }
    return jsonify(response)
