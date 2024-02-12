import pandas as pd
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np

app = Flask(__name__)

# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST']) #POST beause we will give data (from postman)
def predict_api():
    data = request.json(['data']) # whatever input data we are giving, it is in form of json (from postman)
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = regmodel.predict(new_data)
    print(output[1])
    return jsonify(output[0])


@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()] #captures all values from html form
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = regmodel.predict(final_input)[0]
    # print(output[1])
    return render_template("home.html", prediction_text=f'The predicted price is {output}')


if __name__=="__main__":
    app.run(debug = True)