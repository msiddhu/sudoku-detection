from __future__ import division, print_function

from PIL import Image



from codes.solver import getsolution
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer



# Define a flask app
app = Flask(__name__)
"""
# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'
# 
# # Load your trained model
# model = load_model('models/model_resnet.h5')
# #model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('models/model_resnet.h5')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(512, 512))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds
"""


@app.route('/', methods=['GET'])
def index() :
    # Main page
    return render_template('index.html')


@app.route('/siddhu', methods=['GET'])
def siddhu() :
    f = request.data
    print(f)
    return render_template('siddhu.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload() :
    if request.method == 'POST' :
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # preds = model_predict(file_path, model)
        img = Image.open(request.files['file']).convert('RGB')
        image = np.array(img,dtype=np.uint8)
        image = image[:, :, : :-1].copy()
       # img = np.array(img,dtype=np.uint8)
        #img = cv2.resize(img, (512, 512))
        #img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])
        # Convert to string
        # result = getsolution(file_path)

        #result=f.readlines()

        result = getsolution(image)
        return result
       # return  np.array_str(img, precision = 2, suppress_small = True)
    return None


if __name__ == '__main__' :
    app.run(debug=False)
