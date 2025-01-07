
import os
import json
import numpy as np
import argparse
import requests
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite

TARGET_SIZE = (320, 320)
TOP_N_PREDICTIONS=5

def load_image(url):
    response = requests.get(url)
    
    # Check the response status
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type')
        print(f"Content-Type: {content_type}")

        if 'image' in content_type:
            img = Image.open(BytesIO(response.content)) 
            print(f"Loaded image from URL: '{url}'")
        else:
            raise ValueError("The content fetched is not an image.")
    else:
        raise ValueError(f"Error fetching image: {response.status_code}")
    return img
    
    
def initialize_interpreter():
    tflite_model_path = os.environ.get('MODEL_PATH', './classification_model.tflite')    
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()    
    return interpreter
    
    
def load_class_indices():
    #load classes from json file
    classes_json_path=os.environ.get('CLASSES_PATH', './class_indices.json')
    class_indices={}
    with open(classes_json_path, 'r') as json_file:
        class_indices = json.load(json_file)
    return class_indices
    
def prepare_input(url):
    img = load_image(url)
    # Resize the image
    img = img.resize(TARGET_SIZE, Image.NEAREST)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    return preprocess_input(X)    
    
def preprocess_input(x):
        x /= 127.5
        x -= 1.0
        return x    
    
def predict_for_image(interpreter, X):
    input_index = interpreter.get_input_details()[0]['index']
    output_index =interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)    


def decode_predictions(preds, class_indices):
    predictions=preds[0].tolist()
    
    #order in descendent order
    sorted_indices = np.argsort(predictions)[::-1]  

    top_n = -1*TOP_N_PREDICTIONS
    top_indices = np.argsort(predictions)[top_n:][::-1]
    
    # Create a list of tuples for the top predictions
    top_predictions = [(class_label, predictions[index]) for class_label, index in class_indices.items() if index in top_indices]

    # Convert the list of tuples to a dictionary
    top_predictions_dict = {class_label: score for class_label, score in top_predictions}

    return top_predictions_dict
    
def predict(url):
    class_indices = load_class_indices()
    interpreter = initialize_interpreter()
    
    preds = predict_for_image(interpreter, prepare_input(url))
    return decode_predictions(preds, class_indices)
    
    
def lambda_handler(event, context):
    url = event['url']
    
    result = predict(url)
    return result
