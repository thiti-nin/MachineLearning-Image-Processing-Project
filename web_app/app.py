import os
import json
import numpy as np
import cv2
# Tensor
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model("models/classify_model.h5")

# Open json file
with open(os.path.join("food_list/food_list.json"), "r", encoding="utf8") as f:
    food_labels = json.load(f)

print("Model loaded. Check http://127.0.0.1:5000/")

# predict image


def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (224, 224))
    img_tensor = tf.convert_to_tensor(resized_img, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)
    prediction = model.predict(img_tensor, use_multiprocessing=True)
    menu_index = prediction.argmax()
    percent = prediction.max(axis=-1)[0]
    return menu_index, percent

# when user first open webapp


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template('index.html')

# when user click predict button
@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["image"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        pred_label, percent = predict_image(file_path)
        output = food_labels[str(pred_label)]
        # insert percentage to json
        output['percent'] = "{:.2f}".format(percent*100)
        # return json data to front-end
        return output
    return None


if __name__ == "__main__":
    # Serve the app with gevent
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
