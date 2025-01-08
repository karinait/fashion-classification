import os
import json
import numpy as np
import argparse
import requests
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify

# Configuring the Flask application
app = Flask(__name__)

MODELS_DIR = '../models'
MODEL_NAME = 'final_classification_model.keras'
TFLITE_MODEL_NAME = 'classification_model.tflite'
CLASSES_JSON = 'class_indices.json'
TARGET_SIZE = (320, 320)
TOP_N_PREDICTIONS = 5

def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            if 'image' in content_type:
                img = Image.open(BytesIO(response.content))
                print(f"Loaded image from URL: '{image_path_or_url}'")
            else:
                raise ValueError("The content fetched is not an image.")
        else:
            raise ValueError(f"Error fetching image: {response.status_code}")
    else:
        img = Image.open(image_path_or_url)
        print(f"Loaded image from path: '{image_path_or_url}'")
    return img

def initialize_interpreter():
    tflite_model_path = os.environ.get('MODEL_PATH', f'{MODELS_DIR}/classification_model.tflite')    
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()    
    return interpreter
    
    
def load_class_indices():
    #load classes from json file
    classes_json_path=os.environ.get('CLASSES_PATH', f'{MODELS_DIR}/class_indices.json')
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
    
def predict(image_path_or_url):
    """Main function to make a prediction on an image."""
    interpreter = initialize_interpreter()
    class_indices = load_class_indices()
    X = prepare_input(image_path_or_url)
    preds = predict_for_image(interpreter, X)
    return decode_predictions(preds, class_indices)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    image_url = data.get('url')

    if not image_url:
        return jsonify({'error': 'No image URL provided.'}), 400

    try:
        predictions = predict(image_url)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')
    parser.add_argument('image_path_or_url', nargs='?', type=str, help='Path or Url to the image file for prediction')    

    # Parse arguments
    args = parser.parse_args()

    # Prompt for image path if not provided
    if args.image_path_or_url is None:
        image_path_or_url = input("Enter the path to the image or URL: ")
    else:
        image_path_or_url = args.image_path_or_url

    # Validate input before proceeding
    if not image_path_or_url:
        raise ValueError("No image path or URL provided.")

    try:
            predictions = predict(image_path_or_url)
            print("Top Predictions:")
            print(predictions)
    except Exception as e:
            print(f"An error occurred during prediction: {e}")