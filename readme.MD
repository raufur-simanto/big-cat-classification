# A Simple Image Classification Service of Big Cats

## Overview
A service that classifies images of big cats using Deep Learning.

## Features
- Image classification for big cats
- REST API service
- Supports multiple  species of big cats

## Getting Started
### Prerequisites
- Python 3.x
- Flask
- TensorFlow
- Keras
- OpenCV
- Numpy
- Pretrained models for big cats
   -  [VGG16](https://s3.brilliant.com.bd/my_storage/classifier_models/big_cat_classifier.h5
   models)
   -  [MobileNetV2](https://s3.brilliant.com.bd/my_storage/classifier_models/mobilenet_model.h5
   models)

### Installation
1. Clone the repository
    ```
    git clone https://github.com/raufur-simanto/big-cat-classification.git
   
    ```
    ```
    cd big-cat-classifier
    ```
2. Create a virtual environment
    ```
    python3 -m venv venv
    ```
    ```
    source venv/bin/activate
    ```
2. Install the required packages
    ```
    pip install -r requirements.txt
    ```
3. Run the application
    ```
    python app.py
    ```


## Supported Cat Species
- AFRICAN LEOPARD
- CARACAL
- CHEETAH
- CLOUDED LEOPARD
- JAGUAR
- LIONS
- OCELOT
- PUMA
- SNOW 
- LEOPARD
- TIGER


