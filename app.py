from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import requests
import urllib
import logging
import os

app = Flask(__name__)

## set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

## load pretrained model

class_names = ['AFRICAN LEOPARD', 'CARACAL', 'CHEETAH', 'CLOUDED LEOPARD', 'JAGUAR', 'LIONS', 'OCELOT', 'PUMA', 'SNOW LEOPARD', 'TIGER']

def download_and_load_model():
    # model_url = 'https://s3.brilliant.com.bd/my_storage/classifier_models/big_cat_classifier.h5'
    model_url = 'https://s3.brilliant.com.bd/my_storage/classifier_models/mobilenet_model.h5'

    local_model_path = model_url.split('/')[-1]
    print(local_model_path)

    ## Check if the model file already exists locally
    if os.path.exists(local_model_path):
        model = tf.keras.models.load_model(local_model_path)
        return model

    # Download the model file
    response = requests.get(model_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Save the model file locally
    with open(local_model_path, 'wb') as f:
        f.write(response.content)

    # Load the model
    model = tf.keras.models.load_model(local_model_path)
    return model


def load_and_showimg(url):
    img = urllib.request.urlopen(url)
    img_array = np.array(bytearray(img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


@app.route('/classify', methods=['POST'])
def classify_image():
    url = request.json['url']
    try:
        if not url:
            return jsonify({'error': 'No image URL provided'}), 400

        img = load_and_showimg(url)
        model = download_and_load_model()
        prediction = model.predict(img)
        accuracy = float(np.max(prediction))
        logger.info(f"prediction: {prediction}")
        logger.info(f"accuracy: {accuracy}")
        predicted_class = class_names[np.argmax(prediction)]
        logger.info(f"predicted_class: {predicted_class}")
        return jsonify({'prediction': predicted_class, 'accuracy': accuracy}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)