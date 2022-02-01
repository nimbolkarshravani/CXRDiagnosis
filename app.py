from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json
# Flask utils
#.conda\envs\tf\python.exe "G:\BE Project\Web-App\app.py"
from flask import Flask, redirect, url_for, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import json

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the model from external files
# model_NIH_New = Sequential([
#     layers.experimental.preprocessing.Rescaling(1./255, input_shape=(size, size, 1)),
#     layers.Conv2D(16, 3, padding='same', activation='relu',),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),

#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')
# ])

list_of_available_diseases = {
    'Covid19': True, "Atelectasis": True, "Consolidation": False, 
    "Infiltration": False, "Pneumothorax": True, "Edema": False, 
    "Emphysema": False, "Fibrosis": False, "Effusion": True, 
    "Pneumonia": True, "Pleural_thickening": False, "Cardiomegaly": False, 
    "Nodule": True, "Mass": True, "Hernia": False}
# MODEL_ARCHITECTURE = 'cnn_64.json'
# MODEL_WEIGHTS = 'cnn_64.h5'
# json_file = open(MODEL_ARCHITECTURE)
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights(MODEL_WEIGHTS)
# model.summary()
print('Model loaded. Check http://127.0.0.1:5000/')

app = Flask(__name__)                               #Initialize the flask App

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/upload_image',methods=['POST', 'GET'])
def upload_image():
    return render_template('upload_image.html')

@app.route('/testImage',methods=['POST'])
def testImage():
    if request.method == 'POST':
        print("request.form.selectedDiseases: ")
        print(request.form.get("diseasesSelected"))
        diseases_selected = request.form.get("diseasesSelected")
        list_of_selected_diseases = diseases_selected.split(",")
        print("list_of_selected_diseases", list_of_selected_diseases)
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname('./uploads')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        all_disease_predictions = predict_all_diseases(list_of_selected_diseases ,file_path)
        print("Results: ")
        print(all_disease_predictions)
        # preds = model_predict(file_path, model)
        # print(preds)
        # Make prediction
        # preds=preds[0][0]
        # if preds > 0.5:
        #     preds="The Person is Infected With Pneumonia"
        # else:
        #     preds="The Person is NORMAL"
        os.remove(file_path)
        # all_disease_predictions = jsonify(all_disease_predictions)
        # print("Results(JSON): ")
        # print(all_disease_predictions)
    return render_template('report.html', all_disease_predcitions=all_disease_predictions)

def model_predict(img_path, model):
	IMG = image.load_img(img_path).convert('L')
	print(type(IMG))
	# Pre-processing the image
	IMG_ = IMG.resize((64, 64))
	print(type(IMG_))
	IMG_ = np.asarray(IMG_)
	print(IMG_.shape)
	IMG_ = IMG_.reshape((1,64, 64, 1))
	prediction = model.predict(IMG_)
	return prediction

def predict_all_diseases(list_diseases, file_image_path):
    # result = {}
    result = []
    for i in range(len(list_diseases) - 1):
        disease = list_diseases[i]
        if list_of_available_diseases[disease] == True:
            MODEL_ARCHITECTURE = ""
            if disease == "Pneumonia" or disease == "Covid19":
                MODEL_ARCHITECTURE = "models/model_OldModel.json"
            else:
                MODEL_ARCHITECTURE = "models/model_NIH.json"
            MODEL_WEIGHTS = "models/model_" + disease + ".h5"
            json_file = open(MODEL_ARCHITECTURE)
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(MODEL_WEIGHTS)
            model.summary()
            prediction = model_predict(file_image_path, model)
            print("prediction of " + disease)
            print(prediction)
            diseaseResult = int(round(prediction[0][0]*100))
            # result.__setitem__(disease, str((prediction[0][0]*1000)/100))
            result.append(disease)
            result.append(diseaseResult)
    return result

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)